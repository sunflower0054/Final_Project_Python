import cv2
import asyncio
import threading
import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import settings
from services.fall_detector    import process_fall
from services.motion_detector  import process_motion
from services.violent_detector import process_violent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠️  분석할 동영상 파일 경로를 여기에 입력하세요
#     예) VIDEO_PATH = "test.m4v"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO_PATH = "test_video.mp4"   # ← 여기만 바꾸면 됩니다

PORT = 5005
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model     = YOLO("yolov8n.pt")
current_frame  = None
last_sent_date = None  # 마지막으로 daily_activity 전송한 날짜


# ── [SB-013] 자정 전송 함수 ───────────────────────────────────
async def send_daily_activity():
    from services.notifier       import send_daily_score
    import services.motion_detector as motion_detector
    global last_sent_date

    now   = datetime.datetime.now()
    today = datetime.date.today()

    # 자정(00:00~00:00:59) + 오늘 아직 전송 안 했을 때만 실행
    if now.hour == 0 and now.minute == 0 and last_sent_date != today:
        yesterday = today - datetime.timedelta(days=1)
        await send_daily_score(
            date         = str(yesterday),
            motion_score = motion_detector.daily_motion_total
        )
        motion_detector.daily_motion_total = 0  # 누적값 초기화
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
    while True:
        if current_frame is None:
            continue
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


# ── 분석 루프 (웹캠 → 동영상 파일로 변경) ───────────────────────
async def analysis_loop():
    global current_frame

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # main.py 와의 차이점: 웹캠 번호 대신 파일 경로로 열기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ 동영상 파일을 열 수 없습니다: {VIDEO_PATH}")
        print("   VIDEO_PATH 경로를 확인해주세요")
        return

    print(f"✅ 동영상 파일 로드 성공: {VIDEO_PATH}")
    print(f"✅ AI 홈캠 관제 시스템 시작! (동영상 모드, 포트: {PORT})")
    print(f"   스트리밍 주소: http://localhost:{PORT}/video_feed")

    while True:
        ret, frame = cap.read()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # main.py 와의 차이점: 영상 끝나면 처음으로 되감아 반복 재생
        # 발표 중 영상이 멈추는 상황 방지
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not ret:
            print("🔄 영상 끝 → 처음부터 다시 재생")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        fall_result, motion_result, violent_result = await asyncio.gather(
            process_fall(frame),
            process_motion(frame),
            process_violent(frame)
        )

        fallen,          conf_fall    = fall_result
        person_detected, motion_score, last_motion = motion_result
        violent,         person_count, velocity     = violent_result

        # 매 루프마다 자정 체크
        await send_daily_activity()

        results       = yolo_model(frame, verbose=False)
        display_frame = results[0].plot()

        cv2.putText(display_frame,
            f"FALL: {'DETECTED!' if fallen else 'Normal'}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255) if fallen else (0, 255, 0), 2)

        cv2.putText(display_frame,
            f"MOTION: {motion_score} | PERSON: {'YES' if person_detected else 'NO'}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2)

        cv2.putText(display_frame,
            f"VIOLENT: {'DETECTED!' if violent else 'Normal'} | PERSONS: {person_count}",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255) if violent else (255, 165, 0), 2)

        current_frame = display_frame

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