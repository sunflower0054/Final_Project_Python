import cv2
import time
import numpy as np
from ultralytics import YOLO
import settings
# ↑ settings.py에서 no_motion_threshold 값 가져와요

# ── 모델 초기화 ──────────────────────────────────────────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
yolo_model    = YOLO("yolov8n.pt")

# ── 상태 변수 ────────────────────────────────────────────────
no_motion_start_time  = None
event_sent            = False
last_motion_timestamp = None
daily_motion_total = 0  # 오늘 하루 총 픽셀 변화량 누적값

MOTION_PIXEL_MIN = 500
# 픽셀 변화량 기준은 고정값 (슬라이더 대상 아님)

# ── 사람 감지 ────────────────────────────────────────────────
def detect_person(frame):
    results    = yolo_model(frame, verbose=False)
    confidence = 0.0
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                confidence = float(box.conf)
                return True, confidence
    return False, confidence

# ── 픽셀 변화량 계산 ─────────────────────────────────────────
def calculate_motion(frame):
    global daily_motion_total   # ← 이 줄 추가!

    fg_mask      = bg_subtractor.apply(frame)
    motion_score = int(np.sum(fg_mask > 0))

    daily_motion_total += motion_score  # ← 이 줄 추가!

    return motion_score

# ── 무응답 시간 누적 ─────────────────────────────────────────
def accumulate_no_motion(motion_score):
    global no_motion_start_time

    if motion_score < MOTION_PIXEL_MIN:
        if no_motion_start_time is None:
            no_motion_start_time = time.time()

        elapsed = int(time.time() - no_motion_start_time)

        # ↓ 하드코딩 NO_MOTION_THRESHOLD → settings.no_motion_threshold 로 변경!
        if elapsed >= settings.no_motion_threshold:
            return True
    else:
        no_motion_start_time = None

    return False

# ── 무응답 감지 메인 함수 ────────────────────────────────────
async def process_motion(frame):
    global event_sent, last_motion_timestamp

    from services.notifier import send_event

    person_detected, confidence = detect_person(frame)

    if not person_detected:
        return person_detected, 0, last_motion_timestamp

    motion_score = calculate_motion(frame)

    if motion_score >= MOTION_PIXEL_MIN:
        last_motion_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        event_sent = False

    is_confirmed = accumulate_no_motion(motion_score)

    if is_confirmed and not event_sent:
        await send_event(
            event_type = "NO_MOTION_DETECTED",
            frame      = frame,
            confidence = confidence,
            timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata   = {
                "last_motion_timestamp": last_motion_timestamp
            }
        )
        event_sent = True

    return person_detected, motion_score, last_motion_timestamp