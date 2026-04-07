import time
import mediapipe as mp
import settings
# ↑ settings.py에서 fall_sensitivity 값 가져와요
from services.notifier import send_event

# ── MediaPipe 상수 (모델 초기화는 main.py에서, 여기선 landmark 인덱스만 사용) ──
mp_pose = mp.solutions.pose

# ── 상태 변수 ────────────────────────────────────────────────
fall_start_time = None
event_sent      = False
last_event_time  = 0          # ← 추가
COOLDOWN_SECONDS = 60 

FALL_DURATION_THRESHOLD = 10
# 낙상 지속 시간은 고정값 (슬라이더 대상 아님)

# ── 신뢰도 점수 계산 ─────────────────────────────────────────
def get_confidence(landmarks):
    if landmarks is None:
        return 0.0
    nose      = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_hip  = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    confidence = (nose.visibility + left_hip.visibility + right_hip.visibility) / 3
    return round(confidence, 2)

# ── 쓰러진 자세 판단 ─────────────────────────────────────────
def is_fallen(landmarks):
    if landmarks is None:
        return False
    nose      = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_hip  = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_y     = (left_hip.y + right_hip.y) / 2

    if abs(nose.y - hip_y) < settings.fall_sensitivity:
        return True
    return False

# ── 낙상 지속 시간 체크 ──────────────────────────────────────
def check_duration():
    global fall_start_time
    if fall_start_time is None:
        fall_start_time = time.time()
        return False, 0
    elapsed = time.time() - fall_start_time
    if elapsed >= FALL_DURATION_THRESHOLD:
        return True, int(elapsed)
    return False, int(elapsed)

# ── 낙상 감지 메인 함수 ──────────────────────────────────────
# 변경: frame 대신 pose_landmarks를 받아서 처리 (MediaPipe 중복 실행 제거)
async def process_fall(frame, pose_landmarks):
    global fall_start_time, event_sent, last_event_time

    confidence = get_confidence(pose_landmarks)
    fallen     = is_fallen(pose_landmarks)

    now = time.time()

    if fallen:
        is_confirmed, _ = check_duration()
        if is_confirmed and (now - last_event_time > COOLDOWN_SECONDS):  # ← 쿨다운
            send_event(
                event_type = "FALL_DETECTED",
                frame      = frame,
                confidence = confidence,
                timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
                metadata   = {}
            )
            last_event_time = now    # ← 전송 시각 기록
            event_sent = True
    else:
        fall_start_time = None
        event_sent      = False

    return fallen, confidence