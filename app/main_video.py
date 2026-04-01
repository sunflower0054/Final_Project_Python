import cv2
import asyncio
import threading
import datetime
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import uvicorn
import settings
from services.fall_detector    import process_fall
from services.motion_detector  import process_motion
from services.violent_detector import process_violent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠️  분석할 동영상 파일 경로를 여기에 입력하세요
#     예) VIDEO_PATH = "test.m4v"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO_PATH = "app/폭행1.mp4"

PORT = 5005
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO, MediaPipe 여기서 딱 1번만 초기화
# fall_detector, motion_detector, violent_detector 에서
# 중복 초기화 제거 → 프레임당 YOLO 1번, MediaPipe 1번만 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
yolo_model = YOLO("yolov8n.pt")
mp_pose    = mp.solutions.pose

# pose = mp_pose.Pose()
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,             # 1에서 0으로 하향 (가장 가벼운 모델, 저화질에 유리)
    min_detection_confidence=0.2,   # 0.2까지 낮춤 (흐릿해도 일단 시도)
    min_tracking_confidence=0.2     # 추적 감도도 낮춤
)

current_frame  = None
last_sent_date = None  # 마지막으로 daily_activity 전송한 날짜


# ── [SB-013] 자정 전송 함수 ───────────────────────────────────
async def send_daily_activity():
    from services.notifier       import send_daily_score
    import services.motion_detector as motion_detector
    global last_sent_date

    now   = datetime.datetime.now()
    today = datetime.date.today()

    if now.hour == 0 and now.minute == 0 and last_sent_date != today:
        yesterday = today - datetime.timedelta(days=1)
        await send_daily_score(
            date         = str(yesterday),
            motion_score = motion_detector.daily_motion_total
        )
        motion_detector.daily_motion_total = 0
        last_sent_date = today
        print(f"✅ [DAILY] {yesterday} 활동량 전송 완료")


# ── DTO ───────────────────────────────────────────────────────
class AiSettingsDto(BaseModel):
    fall_sensitivity:    float
    no_motion_threshold: int
    velocity_threshold:  float


# ── 스트리밍 제너레이터 ────────────────────────────────────────
def generate_frames():
    global current_frame
    import time
    last_frame = None

    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue

        if current_frame is last_frame:
            time.sleep(0.01)
            continue

        last_frame = current_frame

        _, buffer   = cv2.imencode('.jpg', current_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes +
               b'\r\n')


# ── 스트리밍 엔드포인트 ────────────────────────────────────────
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


# ── AI 감지 설정 엔드포인트 ──────────────────────────────────────
@app.post("/api/settings")
def update_settings(dto: AiSettingsDto):
    settings.fall_sensitivity    = dto.fall_sensitivity
    settings.no_motion_threshold = dto.no_motion_threshold
    settings.velocity_threshold  = dto.velocity_threshold

    print(f"✅ 설정값 업데이트!")
    print(f"   낙상 민감도    : {settings.fall_sensitivity}")
    print(f"   무응답 감지시간: {settings.no_motion_threshold}초")
    print(f"   폭행 임계값    : {settings.velocity_threshold}")

    return {
        "success": True,
        "fall_sensitivity":    settings.fall_sensitivity,
        "no_motion_threshold": settings.no_motion_threshold,
        "velocity_threshold":  settings.velocity_threshold
    }

@app.get("/api/settings")
def get_settings():
    return {
        "success": True,
        "fall_sensitivity":    settings.fall_sensitivity,
        "no_motion_threshold": settings.no_motion_threshold,
        "velocity_threshold":  settings.velocity_threshold
    }


# ── 분석 루프 (동영상 파일) ───────────────────────────────────
async def analysis_loop():
    global current_frame

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ 동영상 파일을 열 수 없습니다: {VIDEO_PATH}")
        print("   VIDEO_PATH 경로를 확인해주세요")
        return

    print(f"✅ 동영상 파일 로드 성공: {VIDEO_PATH}")
    print(f"✅ AI 홈캠 관제 시스템 시작! (동영상 모드, 포트: {PORT})")
    print(f"   스트리밍 주소: http://localhost:{PORT}/video_feed")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 프레임 스킵: 화면은 매 프레임 갱신,
    # AI 분석은 SKIP_INTERVAL 프레임마다 1번만 실행
    # 권장: 3 (발표용) / 1 (정확도 최우선)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SKIP_INTERVAL = 3
    frame_count   = 0

    while True:
        ret, frame = cap.read()

        # 영상 끝나면 처음으로 되감아 반복 재생 (발표 중 멈춤 방지)
        if not ret:
            print("🔄 영상 끝 → 처음부터 다시 재생")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             # 상태 초기화
            import services.fall_detector as fd
            import services.motion_detector as md
            import services.violent_detector as vd
            fd.fall_start_time = None;  fd.event_sent = False
            md.no_motion_start_time = None; md.event_sent = False
            vd.consecutive_count = 0;  vd.event_sent = False
            continue

        await asyncio.sleep(0.03)   # 30ms 대기, 이벤트 루프 양보 O

        frame_count += 1

        # 화면은 매 프레임 갱신 (스트리밍 끊김 없음)
        current_frame = frame

        # AI 분석은 SKIP_INTERVAL 프레임마다 1번만
        if frame_count % SKIP_INTERVAL != 0:
            continue
    
        yolo_results   = yolo_model(frame, verbose=False)
        rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result    = pose.process(rgb_frame)
        pose_landmarks = pose_result.pose_landmarks

        # ── 바운딩박스 그리기 (사람만, 텍스트 없이) ──────────────
        annotated_frame = frame.copy()
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls) == 0:                          # 사람(class 0)만
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                          (0, 0, 178), 2)              # 초록색, 두께 2

        current_frame = annotated_frame                        # ← frame 대신 annotated_frame
# ─────────────────────────────────────────────────────

        fall_result, motion_result, violent_result = await asyncio.gather(
            process_fall(frame, pose_landmarks),
            process_motion(frame, yolo_results),
            process_violent(frame, yolo_results, pose_landmarks)
        )

        fallen, conf_fall     = fall_result
        person_detected, motion_score, last_motion = motion_result
        violent, person_count, velocity             = violent_result

        # 디버깅은 여기서
        print(f"🔍 감지 물체: {len(yolo_results[0].boxes)}개 | 뼈대: {'✅' if pose_landmarks else '❌'}")
        print(f"📊 낙상:{fallen}, 활동:{motion_score:.0f}, 폭행:{violent}")

        await send_daily_activity()

    cap.release()


# ── 분석 루프 별도 스레드 실행 ────────────────────────────────
def run_analysis():
    asyncio.run(analysis_loop())

if __name__ == "__main__":
    thread        = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
    # http://localhost:5005/video_feed 브라우저에 입력!