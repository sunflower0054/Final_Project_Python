import cv2
import time
import numpy as np
import settings
# ↑ settings.py에서 no_motion_threshold 값 가져와요
from services.notifier import send_event

# ── 모델 초기화 (MOG2만 여기서, YOLO는 main.py에서) ──────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# ── 상태 변수 ────────────────────────────────────────────────
no_motion_start_time  = None
event_sent            = False
last_motion_timestamp = None
daily_motion_total    = 0  # 오늘 하루 총 픽셀 변화량 누적값
last_event_time  = 0          # ← 추가
COOLDOWN_SECONDS = 60  

MOTION_PIXEL_MIN = 500
# 픽셀 변화량 기준은 고정값 (슬라이더 대상 아님)

# ── 사람 감지 (YOLO 결과를 받아서 파싱, 중복 실행 제거) ──────
def detect_person_from_results(yolo_results):
    confidence = 0.0
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls) == 0:
                confidence = float(box.conf)
                return True, confidence
    return False, confidence

# ── 픽셀 변화량 계산 ─────────────────────────────────────────
def calculate_motion(frame):
    global daily_motion_total

    fg_mask      = bg_subtractor.apply(frame)
    motion_score = int(np.sum(fg_mask > 0))

    daily_motion_total += motion_score

    return motion_score

# ── 무응답 시간 누적 ─────────────────────────────────────────
def accumulate_no_motion(motion_score):
    global no_motion_start_time

    if motion_score < MOTION_PIXEL_MIN:
        if no_motion_start_time is None:
            no_motion_start_time = time.time()

        elapsed = int(time.time() - no_motion_start_time)

        if elapsed >= settings.no_motion_threshold:
            return True
    else:
        no_motion_start_time = None

    return False

# ── 무응답 감지 메인 함수 ────────────────────────────────────
# 변경: yolo_results를 받아서 처리 (YOLO 중복 실행 제거)
async def process_motion(frame, yolo_results):
    global event_sent, last_motion_timestamp, last_event_time

    person_detected, confidence = detect_person_from_results(yolo_results)

    if not person_detected:
        return person_detected, 0, last_motion_timestamp

    motion_score = calculate_motion(frame)

    if motion_score >= MOTION_PIXEL_MIN:
        last_motion_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        event_sent = False

    is_confirmed = accumulate_no_motion(motion_score)
    now = time.time()

    if is_confirmed and (now - last_event_time > COOLDOWN_SECONDS):  # ← 쿨다운
        await send_event(
            event_type = "NO_MOTION_DETECTED",
            frame      = frame,
            confidence = confidence,
            timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata   = {"last_motion_timestamp": last_motion_timestamp}
        )
        last_event_time = now    # ← 전송 시각 기록
        event_sent = True

    return person_detected, motion_score, last_motion_timestamp