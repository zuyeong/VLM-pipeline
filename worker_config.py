"""
SSH 처리 서버 측 워커(ssh_server.py) 설정.

이 파일은 워커가 도는 호스트 — 공개 서버가 SSH 로 접속해 들어오는
그 처리 서버 — 에 배포된다. 공개 서버의 config.py 와는 다른 머신에 있고,
역할도 정반대(공개 서버: SSH 클라이언트 / 처리 서버: 파일 시스템 + GPU)
이므로 두 config 를 섞지 않는다.

배포 전, 아래 ★ 표시된 항목을 채우고 ssh_server.py 를 띄운다.
"""
from __future__ import annotations

# ── 디렉토리 경로 ─────────────────────────────────────────────────────────
# ★ 절대 경로로 적는다. 공개 서버 config.py 의 REMOTE_INBOX_DIR /
#   REMOTE_OUTBOX_DIR 값과 1:1 일치해야 한다 (=같은 디렉토리를 양쪽이 봄).
#   WORK 는 워커 내부 작업용 — 공개 서버는 모름. INBOX/OUTBOX 와 같은
#   파일시스템(같은 마운트)에 두면 결과 이미지 이동이 원자적이라 안전.
INBOX_DIR: str = "/home/cjy/workspace/pipeline/Input"      # ★ 공개 서버가 작업을 떨어뜨리는 곳
OUTBOX_DIR: str = "/home/cjy/workspace/pipeline/Output"    # ★ 워커가 결과를 떨어뜨리는 곳
WORK_DIR: str = "/home/cjy/workspace/pipeline/Work"        # ★ 처리 중 임시 디렉토리

# ── 폴링 ──────────────────────────────────────────────────────────────────
# INBOX 에 새 작업이 들어왔는지 확인하는 주기(초). 너무 짧으면 디스크 부담,
# 너무 길면 사용자 체감 지연 증가. 1초가 합리적 기본값.
POLL_INTERVAL_SEC: float = 1.0

# ── 로그 ──────────────────────────────────────────────────────────────────
# DEBUG / INFO / WARNING / ERROR 중 하나. 운영에서는 INFO 권장.
LOG_LEVEL: str = "INFO"

# ── 백본(pipeline) 통합 ────────────────────────────────────────────────────
# ssh_server.py 가 import 할 백본 폴더. 워커 부팅 시 sys.path 앞에 추가한다.
PIPELINE_ROOT: str = "/home/cjy/workspace/pipeline"

# 백본 모델 선택용 config.yaml 경로. PIPELINE_CONFIG_OVERRIDE 가 None 일 때 사용.
PIPELINE_CONFIG_PATH: str = "/home/cjy/workspace/pipeline/config.yaml"

# 테스트성 오버라이드. dict 면 그 값을 그대로 사용 (config.yaml 무시).
# 예: PIPELINE_CONFIG_OVERRIDE = {"models": {"segmenter": {"name": "dummy_seg"},
#                                            "inpainter": {"name": "dummy_inpaint"}}}
# 운영에선 None 유지.
PIPELINE_CONFIG_OVERRIDE = None
