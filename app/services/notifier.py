import httpx
import cv2
import time

SPRING_BOOT_ENABLED = True  # Spring Boot 연동 시 True로 변경
SPRING_BOOT_URL = "http://localhost:8091/api/v1/events/receive"  
SPRING_BOOT_DAILY_URL = "http://localhost:8091/api/v1/daily-activity"
RESIDENT_ID = 22
MAX_RETRY = 3
RETRY_DELAYS = [1, 5, 30]

def frame_to_bytes(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def send_event(event_type, frame, confidence, timestamp, metadata=None):
    if not SPRING_BOOT_ENABLED:
        print(f"✅ [{event_type}] 감지됨! (전송 비활성화 상태)")
        return

    image_bytes = frame_to_bytes(frame)

    payload = {
        "resident_id": str(RESIDENT_ID),
        "event_type":  event_type,
        "timestamp":   timestamp,
        "confidence":  str(confidence),
        "metadata":    str(metadata or {})
    }

    for attempt in range(MAX_RETRY):
        try:
            with httpx.Client() as client:
                response = client.post(
                    SPRING_BOOT_URL,
                    data=payload,
                    files={"frame_image": ("capture.jpg", image_bytes, "image/jpeg")},
                    timeout=10.0
                )

            if response.status_code == 200:
                print(f"✅ [{event_type}] 이벤트 전송 성공")
                return
            else:
                print(f"⚠️ [{event_type}] 전송 실패 (상태코드: {response.status_code}), {attempt+1}번째 시도")

        except Exception as e:
            print(f"❌ [{event_type}] 전송 오류: {e}, {attempt+1}번째 시도")

        if attempt < MAX_RETRY - 1:
            time.sleep(RETRY_DELAYS[attempt])

    print(f"🚨 [{event_type}] 이벤트 전송 최종 실패 (3회 모두 실패)")


def send_daily_score(date: str, motion_score: int):
    if not SPRING_BOOT_ENABLED:
        print(f"✅ [DAILY] {date} motion_score={motion_score} (전송 비활성화)")
        return

    payload = {
        "resident_id":  str(RESIDENT_ID),
        "date":         date,
        "motion_score": str(motion_score)
    }

    for attempt in range(MAX_RETRY):
        try:
            with httpx.Client() as client:
                response = client.post(
                    SPRING_BOOT_DAILY_URL,
                    data=payload,
                    timeout=10.0
                )

            if response.status_code == 200:
                print(f"✅ [DAILY] 활동량 전송 완료: {date} = {motion_score}")
                return
            else:
                print(f"⚠️ [DAILY] 전송 실패 (상태코드: {response.status_code}), {attempt+1}번째 시도")

        except Exception as e:
            print(f"❌ [DAILY] 전송 오류: {e}, {attempt+1}번째 시도")

        if attempt < MAX_RETRY - 1:
            time.sleep(RETRY_DELAYS[attempt])

    print(f"🚨 [DAILY] 활동량 전송 최종 실패 (3회 모두 실패) — {date} = {motion_score}")