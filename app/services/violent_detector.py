import time
import numpy as np
import mediapipe as mp
import settings
# ↑ settings.py에서 velocity_threshold 값 가져와요
from services.notifier import send_event

# ── MediaPipe 상수 (모델 초기화는 main.py에서, 여기선 landmark 인덱스만 사용) ──
mp_pose = mp.solutions.pose

# ── 상태 변수 ────────────────────────────────────────────────
prev_keypoints    = None
consecutive_count = 0
event_sent        = False
last_event_time   = 0         # ← 추가
COOLDOWN_SECONDS  = 30        # ← 같은 이벤트 재전송 최소 간격

CONSECUTIVE_FRAMES = 3
# 연속 프레임 기준은 고정값 (슬라이더 대상 아님)

# ── 인물 수 파악 (YOLO 결과를 받아서 파싱, 중복 실행 제거) ──
def count_persons_from_results(yolo_results):
    count      = 0
    total_conf = 0.0
    for result in yolo_results:
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
# 변경: yolo_results, pose_landmarks를 받아서 처리 (YOLO, MediaPipe 중복 실행 제거)
async def process_violent(frame, yolo_results, pose_landmarks):
    global prev_keypoints, event_sent, last_event_time

    person_count, confidence = count_persons_from_results(yolo_results)
    velocity                 = calculate_velocity(prev_keypoints, pose_landmarks)
    violent_flag             = is_violent(velocity, person_count)
    is_confirmed             = check_consecutive_frames(violent_flag)

    now = time.time()

    if is_confirmed and (now - last_event_time > COOLDOWN_SECONDS):  # ← 쿨다운 체크
        send_event(
            event_type = "VIOLENT_MOTION_DETECTED",
            frame      = frame,
            confidence = confidence,
            timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata   = {
                "person_count": person_count,
                "max_velocity": velocity
            }
        )
        last_event_time = now    # ← 전송 시각 기록
        event_sent = True

    if not violent_flag:
        event_sent = False

    prev_keypoints = pose_landmarks
    return violent_flag, person_count, velocity