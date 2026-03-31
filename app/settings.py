# app/settings.py
# 모든 detector가 이 값을 공유해서 참조해요
# Spring Boot가 /api/settings로 POST하면 이 값들이 업데이트돼요

# fall_sensitivity     = 0.1    # 낙상 감지 민감도 (nose-hip y좌표 차이 임계값)
# no_motion_threshold  = 1800   # 무응답 감지 시간 (초) — 기본 30분
# velocity_threshold   = 0.15   # 폭행 의심 임계값 (관절 이동속도)


# 발표용 settings.py
fall_sensitivity     = 0.3    # 0.1 → 0.3 (더 쉽게 낙상 판정)
no_motion_threshold  = 5     # 1800 → 5  (5초 무움직임이면 감지)
velocity_threshold   = 0.07   # 0.15 → 0.07 (더 느린 움직임도 폭행 감지)


# fall_sensitivity 0.1 → 0.3
# 코와 골반의 y좌표 차이 임계값. 0.1이면 거의 완전히 수평으로 누워야 감지되는데, 0.3이면 약간만 기울어져도 감지.
# no_motion_threshold 1800 → 10
# 30분을 5초로 줄임. 
# velocity_threshold 0.15 → 0.08
# 관절 이동속도 임계값. 0.15면 꽤 빠르게 움직여야 하는데, 0.07면 일반적인 움직임도 폭행 의심으로 잡힘. 