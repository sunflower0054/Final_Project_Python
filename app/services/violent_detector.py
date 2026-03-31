import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import settings
# ↑ settings.py에서 velocity_threshold 값 가져와요

# ── 모델 초기화 ──────────────────────────────────────────────
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose()
yolo_model = YOLO("yolov8n.pt")

# ── 상태 변수 ────────────────────────────────────────────────
prev_keypoints    = None
consecutive_count = 0
event_sent        = False

CONSECUTIVE_FRAMES = 3
# 연속 프레임 기준은 고정값 (슬라이더 대상 아님)

# ── 관절 좌표 추출 ───────────────────────────────────────────
def extract_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = pose.process(rgb_frame)
    return result.pose_landmarks

# ── 인물 수 파악 ─────────────────────────────────────────────
def count_persons(frame):
    results    = yolo_model(frame, verbose=False)
    count      = 0
    total_conf = 0.0
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                count      += 1
                total_conf += float(box.conf)
    avg_confidence = round(total_conf / count, 2) if count > 0 else 0.0
    return count, avg_confidence

# ── 관절 이동 속도 계산 ──────────────────────────────────────
def calculate_velocity(prev_landmarks, curr_landmarks):
    if prev_landmarks is None or curr_landmarks is None:
        return 0.0
    velocities = []
    for i in range(len(curr_landmarks.landmark)):
        prev = prev_landmarks.landmark[i]
        curr = curr_landmarks.landmark[i]
        velocity = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
        velocities.append(velocity)
    return round(float(np.max(velocities)), 2)

# ── 폭행 여부 판단 ───────────────────────────────────────────
def is_violent(velocity, person_count):
    # ↓ 하드코딩 0.15 → settings.velocity_threshold 로 변경!
    return person_count >= 2 and velocity >= settings.velocity_threshold

# ── 연속 프레임 체크 ─────────────────────────────────────────
def check_consecutive_frames(is_violent_flag):
    global consecutive_count
    if is_violent_flag:
        consecutive_count += 1
    else:
        consecutive_count = 0
    return consecutive_count >= CONSECUTIVE_FRAMES

# ── 폭행 감지 메인 함수 ──────────────────────────────────────
async def process_violent(frame):
    global prev_keypoints, event_sent
    from services.notifier import send_event

    person_count, confidence = count_persons(frame)
    curr_keypoints           = extract_keypoints(frame)
    velocity                 = calculate_velocity(prev_keypoints, curr_keypoints)
    violent_flag             = is_violent(velocity, person_count)
    is_confirmed             = check_consecutive_frames(violent_flag)

    if is_confirmed and not event_sent:
        await send_event(
            event_type = "VIOLENT_MOTION_DETECTED",
            frame      = frame,
            confidence = confidence,
            timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata   = {
                "person_count": person_count,
                "max_velocity": velocity
            }
        )
        event_sent = True

    if not violent_flag:
        event_sent = False

    prev_keypoints = curr_keypoints
    return violent_flag, person_count, velocity