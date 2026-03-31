import cv2
import time
import mediapipe as mp
import settings
# ↑ settings.py에서 fall_sensitivity 값 가져와요

# ── 모델 초기화 ──────────────────────────────────────────────
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose()

# ── 상태 변수 ────────────────────────────────────────────────
fall_start_time = None
event_sent      = False

FALL_DURATION_THRESHOLD = 10
# 낙상 지속 시간은 고정값 (슬라이더 대상 아님)

# ── 관절 좌표 추출 ───────────────────────────────────────────
def extract_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = pose.process(rgb_frame)
    return result.pose_landmarks

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

    # ↓ 하드코딩 0.1 → settings.fall_sensitivity 로 변경!
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
async def process_fall(frame):
    global fall_start_time, event_sent
    from services.notifier import send_event

    landmarks  = extract_keypoints(frame)
    confidence = get_confidence(landmarks)
    fallen     = is_fallen(landmarks)

    if fallen:
        is_confirmed, _ = check_duration()
        if is_confirmed and not event_sent:
            await send_event(
                event_type = "FALL_DETECTED",
                frame      = frame,
                confidence = confidence,
                timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S"),
                metadata   = {}
            )
            event_sent = True
    else:
        fall_start_time = None
        event_sent      = False

    return fallen, confidence