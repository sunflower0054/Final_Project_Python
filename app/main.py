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
pose       = mp_pose.Pose()

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


# ── 분석 루프 ─────────────────────────────────────────────────
async def analysis_loop():
    global current_frame

    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        if ret:
            print(f"✅ {i}번 웹캠 연결 성공!")
            break
        cap.release()

    if not ret:
        print("❌ 웹캠을 열 수 없습니다")
        return

    print("✅ AI 홈캠 관제 시스템 시작!")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 프레임 스킵: 화면은 매 프레임 갱신,
    # AI 분석은 SKIP_INTERVAL 프레임마다 1번만 실행
    # 권장: 3 (발표용) / 1 (정확도 최우선)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SKIP_INTERVAL = 3
    frame_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 화면은 매 프레임 갱신 (스트리밍 끊김 없음)
        current_frame = frame

        # AI 분석은 SKIP_INTERVAL 프레임마다 1번만
        if frame_count % SKIP_INTERVAL != 0:
            continue

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # YOLO, MediaPipe 각 1번씩만 실행 후 결과 공유
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        yolo_results   = yolo_model(frame, verbose=False)
        rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result    = pose.process(rgb_frame)
        pose_landmarks = pose_result.pose_landmarks

        # 각 detector에 결과 전달 (중복 실행 없음)
        fall_result, motion_result, violent_result = await asyncio.gather(
            process_fall(frame, pose_landmarks),
            process_motion(frame, yolo_results),
            process_violent(frame, yolo_results, pose_landmarks)
        )

        fallen,          conf_fall    = fall_result
        person_detected, motion_score, last_motion = motion_result
        violent,         person_count, velocity     = violent_result

        await send_daily_activity()

    cap.release()


# ── 분석 루프 별도 스레드 실행 ────────────────────────────────
def run_analysis():
    asyncio.run(analysis_loop())

if __name__ == "__main__":
    thread        = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=5005)
    # http://localhost:5005/video_feed 브라우저에 입력!