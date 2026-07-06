"""
서브프로세스 인페인터 공용 헬퍼.

각 인페인터 (lama / sd / mat) 가 conda env 활성화 후 외부 추론 스크립트를
실행하는 동일 패턴을 공유한다. 이 모듈은 그 패턴의 두 가지 책임을 일원화:

  1. PID + UUID 기반 고유 임시 디렉토리 — 동시 호출 시 충돌 방지.
  2. 서브프로세스 stderr/stdout 캡처 — 실패 시 진짜 원인을 호출자에게 전달.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from typing import Iterator


# ┌─ 정책: 모든 임시 파일은 /home/cjy/workspace/ 안에만 (사용자 요청). ──────┐
# │ 환경변수 PIPELINE_TMP_BASE 로 오버라이드 가능. 부재 시 workspace 안 tmp. │
# └─────────────────────────────────────────────────────────────────────────┘
DEFAULT_TMP_BASE = os.environ.get(
    "PIPELINE_TMP_BASE",
    "/home/cjy/workspace/pipeline/tmp",
)


@contextmanager
def unique_tmp_dir(label: str, base: str | None = None) -> Iterator[str]:
    """동시 호출 시 충돌하지 않는 고유 임시 디렉토리를 만들고 정리한다.

    이름 규칙: <base>/<label>_<pid>_<uuid8>/
    예) /home/cjy/workspace/pipeline/tmp/lama_inpaint_12345_a1b2c3d4/

    base 가 None 이면 DEFAULT_TMP_BASE (workspace 안 tmp) 를 사용.
    base 디렉토리가 없으면 자동 생성.
    finally 블록에서 항상 삭제 — 호출자가 예외로 빠져나가도 누수 없음.
    """
    base = base or DEFAULT_TMP_BASE
    os.makedirs(base, exist_ok=True)  # base 자체 보장
    name = f"{label}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=False)  # 이름이 겹칠 가능성 사실상 0
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run_bash(cmd: str, label: str) -> None:
    """bash 로 명령 실행, 실패 시 stderr/stdout 꼬리를 포함한 RuntimeError 로 보고.

    기존 패턴 (`subprocess.run(..., check=True)`) 은 returncode 만 떨어뜨려 실제
    원인 (CUDA OOM / 파일 없음 / 권한 부족 등) 을 알 수 없었다. capture_output
    으로 둘 다 잡아두고 마지막 수 KB 를 메시지에 포함한다.

    ⚠ 메모 B-8 (capture_output 메모리 폭발 가능 시나리오 — 일단 패스, 추후 처리):
        capture_output=True 는 stdout/stderr 를 전부 메모리에 적재한다. 정상
        운영에선 KB 단위라 안전하지만 아래 비정상 시나리오에서는 GB 단위 폭발 가능:
          1) 모델이 tqdm 류 progress bar 를 stderr 에 매초 갱신하며 무한 루프
          2) CUDA error retry 가 같은 trace 를 무한 반복 출력
          3) huggingface hub 다운로드 fail → 진행률 % 가 MB 단위로 폭주
          4) flash-attn 의 verbose 모드가 토큰별 로그 출력
        근본 해결: Popen + 스트리밍 read + 마지막 N KB 만 유지하는 ring buffer.
        지금은 운영 빈도 낮으므로 미적용.
    """
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        stdout_tail = (result.stdout or "")[-1000:]
        raise RuntimeError(
            f"[{label}] subprocess failed (exit={result.returncode})\n"
            f"--- stderr (tail) ---\n{stderr_tail}\n"
            f"--- stdout (tail) ---\n{stdout_tail}"
        )
