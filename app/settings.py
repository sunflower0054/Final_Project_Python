# app/settings.py
# 모든 detector가 이 값을 공유해서 참조해요
# Spring Boot가 /api/settings로 POST하면 이 값들이 업데이트돼요

fall_sensitivity     = 0.1    # 낙상 감지 민감도 (nose-hip y좌표 차이 임계값)
no_motion_threshold  = 1800   # 무응답 감지 시간 (초) — 기본 30분
velocity_threshold   = 0.15   # 폭행 의심 임계값 (관절 이동속도)