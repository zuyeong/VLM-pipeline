"""
VisionReasoner segmenter — 데몬 클라이언트.

메인 워커 conda env (예: wise) 안에서 도는 어댑터. 별도 conda env
(visionreasoner_test) 의 데몬 프로세스를 spawn 해서 JSON IPC 로 통신.

설계 메모:
  - 데몬은 워커 lifetime 동안 1번만 부팅 (cold ~5분 흡수).
  - 매 predict() 호출당 inference 만 (~10-30s).
  - 데몬 사망 감지 시 자동 1회 재시작 — 그래도 실패하면 호출자에게 raise.
  - thread lock — 동시 predict() 호출 직렬화.
  - select 기반 timeout — 데몬 hang 감지.
  - stderr 는 파일로 redirect — pipe 차서 데몬 멈추는 것 방지.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional

from PIL import Image

from core.base_models import BaseSegmentationModel


# 기본값들. 모든 산출물은 /home/cjy/workspace/ 안에 둠 (사용자 정책).
_DEFAULT_STARTUP_TIMEOUT_SEC = 600   # VR 모델 로드 최대 10분 허용
_DEFAULT_PREDICT_TIMEOUT_SEC = 180   # 한 inference 최대 3분
_DEFAULT_WORKSPACE_TMP = "/home/cjy/workspace/pipeline/tmp"
_DEFAULT_LOG_DIR = "/home/cjy/workspace/pipeline/logs"


class VisionReasonerSegmentationModel(BaseSegmentationModel):
    """
    데몬 패턴 segmenter.

    reads (from ctx in pipeline): image, prompt
    writes: mask (PIL "L")

    워커 부팅 흐름:
      1. load_model() 호출
      2. 데몬 spawn (subprocess.Popen)
      3. stdout 첫 줄 "READY" 까지 대기 (timeout: startup_timeout_sec)
      4. 정상이면 predict() 받을 준비 완료

    각 predict() 흐름:
      1. proc.poll() — 데몬 살아있는지 확인. 죽었으면 1회 재시작 시도.
      2. 임시 PNG 저장 (입력)
      3. JSON request line 송신
      4. select() 로 timeout 두고 response line readline
      5. response 의 output_mask_path 에서 PNG 읽어 PIL "L" 반환
      6. 임시 파일 정리
    """

    def __init__(
        self,
        conda_env_python: str,
        vr_root: str,
        daemon_script_path: str,
        model_path: str = "pretrained_models/VisionReasoner-7B",
        task_router_path: str = "pretrained_models/TaskRouter-1.5B",
        segmentation_model_path: str = "facebook/sam2-hiera-large",
        device: str = "cuda:0",
        force_task: str = "segmentation",
        startup_timeout_sec: int = _DEFAULT_STARTUP_TIMEOUT_SEC,
        predict_timeout_sec: int = _DEFAULT_PREDICT_TIMEOUT_SEC,
        daemon_log_path: Optional[str] = None,
    ):
        self.conda_env_python = conda_env_python
        self.vr_root = vr_root
        self.daemon_script_path = daemon_script_path
        self.model_path = model_path
        self.task_router_path = task_router_path
        self.segmentation_model_path = segmentation_model_path
        self.device = device
        self.force_task = force_task
        self.startup_timeout_sec = startup_timeout_sec
        self.predict_timeout_sec = predict_timeout_sec
        # 로그는 workspace 안 logs/ 폴더. PID 포함해 다른 워커들과 로그 충돌 방지.
        os.makedirs(_DEFAULT_LOG_DIR, exist_ok=True)
        self.daemon_log_path = daemon_log_path or os.path.join(
            _DEFAULT_LOG_DIR, f"vr_daemon_{os.getpid()}.log"
        )

        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._lock = threading.Lock()
        self._tmp_dir: Optional[str] = None
        self._restart_count = 0

    # ── 로그 (메인 워커 측) ─────────────────────────────────────────
    def _print(self, msg: str) -> None:
        print(f"[VR-Adapter] {msg}", flush=True)

    # ── 데몬 lifecycle ─────────────────────────────────────────────
    def load_model(self, **kwargs) -> None:
        """워커 부팅 시 1회 호출. 데몬 spawn + READY 대기."""
        # 경로 검증.
        for label, path in (
            ("conda_env_python", self.conda_env_python),
            ("vr_root", self.vr_root),
            ("daemon_script_path", self.daemon_script_path),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} 경로 없음: {path}")

        # 임시 디렉토리 — predict() 의 입력/출력 교환소. 정책상 workspace 안에만.
        os.makedirs(_DEFAULT_WORKSPACE_TMP, exist_ok=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="vr_seg_", dir=_DEFAULT_WORKSPACE_TMP)

        self._spawn_daemon()

    def _spawn_daemon(self) -> None:
        # 이전 데몬 정리.
        self._kill_daemon()

        cmd = [
            self.conda_env_python,
            self.daemon_script_path,
            "--vr-root", self.vr_root,
            "--model-path", self.model_path,
            "--task-router-path", self.task_router_path,
            "--segmentation-model-path", self.segmentation_model_path,
            "--device", self.device,
        ]
        self._print(f"spawning daemon: {' '.join(cmd)}")
        self._print(f"daemon stderr → {self.daemon_log_path}")

        # stderr 파일로 redirect — pipe 가 차서 데몬이 멈추는 사고 방지.
        self._log_file = open(self.daemon_log_path, "ab")

        env = os.environ.copy()
        # CUDA_VISIBLE_DEVICES 는 데몬이 자체적으로 args.device 보고 설정.

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            bufsize=1,
            text=True,
            env=env,
        )

        t_start = time.time()
        self._print(f"daemon pid={self._proc.pid}, waiting READY (timeout={self.startup_timeout_sec}s)...")

        # READY 대기.
        deadline = time.time() + self.startup_timeout_sec
        while time.time() < deadline:
            # 죽었으면 즉시 실패.
            if self._proc.poll() is not None:
                code = self._proc.returncode
                raise RuntimeError(
                    f"VR daemon died during startup (exit={code}). "
                    f"see {self.daemon_log_path}"
                )

            # stdout 한 줄 시도.
            line = self._proc.stdout.readline()
            if not line:
                # EOF 또는 일시적 빈 라인 — poll 로 살아있나 재확인.
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"VR daemon stdout EOF during startup "
                        f"(exit={self._proc.returncode}). see {self.daemon_log_path}"
                    )
                continue

            line = line.strip()
            if line == "READY":
                elapsed = time.time() - t_start
                self._print(f"daemon READY (took {elapsed:.1f}s)")
                return

            # JSON fatal 인지 확인.
            try:
                payload = json.loads(line)
                if payload.get("fatal"):
                    err = payload.get("error", "unknown")
                    raise RuntimeError(
                        f"VR daemon reported fatal error during startup: {err}"
                    )
            except json.JSONDecodeError:
                # 알 수 없는 라인 — 데몬 stdout 채널 오염 가능. 경고 후 계속.
                self._print(f"unexpected daemon stdout: {line[:200]}")

        # timeout.
        self._kill_daemon()
        raise RuntimeError(
            f"VR daemon did not become READY within {self.startup_timeout_sec}s. "
            f"see {self.daemon_log_path}"
        )

    def _kill_daemon(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                # stdin close — for-line-in-sys.stdin 루프가 빠져나가게.
                if self._proc.stdin is not None:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:
                pass
        self._proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    # ── BaseSegmentationModel 인터페이스 ─────────────────────────
    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        with self._lock:
            return self._predict_once(image, prompt)

    def _predict_once(self, image: Image.Image, prompt: str) -> Image.Image:
        # 데몬 헬스 체크 + 죽었으면 1회 재시작.
        if self._proc is None or self._proc.poll() is not None:
            code = self._proc.returncode if self._proc else None
            self._print(f"daemon dead (exit={code}) — restarting (cold start)")
            self._restart_count += 1
            self._spawn_daemon()

        assert self._tmp_dir is not None

        task_id = uuid.uuid4().hex[:12]
        in_png = os.path.join(self._tmp_dir, f"{task_id}_in.png")
        out_mask = os.path.join(self._tmp_dir, f"{task_id}_mask.png")

        try:
            image.save(in_png)

            req = {
                "task_id": task_id,
                "image_path": in_png,
                "prompt": prompt,
                "output_mask_path": out_mask,
                "task": self.force_task,
            }
            req_line = json.dumps(req, ensure_ascii=False) + "\n"

            # 송신.
            self._proc.stdin.write(req_line)
            self._proc.stdin.flush()

            # 응답 대기 — select 로 timeout 적용 (Linux only).
            stdout_fd = self._proc.stdout.fileno()
            ready, _, _ = select.select([stdout_fd], [], [], self.predict_timeout_sec)
            if not ready:
                # hang — 데몬이 무한 루프 또는 멈춤. 죽이고 raise.
                self._print(f"daemon predict timeout ({self.predict_timeout_sec}s) — killing")
                self._kill_daemon()
                raise RuntimeError(
                    f"VR daemon predict timeout ({self.predict_timeout_sec}s) on task {task_id}"
                )

            response_line = self._proc.stdout.readline()
            if not response_line:
                code = self._proc.poll()
                raise RuntimeError(
                    f"VR daemon stdout EOF (exit={code}) on task {task_id}. "
                    f"see {self.daemon_log_path}"
                )

            try:
                resp = json.loads(response_line.strip())
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"VR daemon returned non-JSON: {response_line[:200]!r} ({e})"
                )

            if not resp.get("done"):
                err = resp.get("error", "unknown")
                raise RuntimeError(f"VR daemon error on task {task_id}: {err}")

            mask_path = resp.get("output_mask_path") or out_mask
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"VR daemon reported done but mask file not found: {mask_path}"
                )

            mask = Image.open(mask_path).convert("L")
            mask.load()  # 임시 파일 삭제 전에 메모리에 완전 로드
            elapsed = resp.get("elapsed_sec", -1)
            num_objects = resp.get("num_objects", -1)
            self._print(
                f"task {task_id} done in {elapsed:.1f}s, objects={num_objects}, "
                f"mask={mask.size}"
            )
            return mask
        finally:
            for p in (in_png, out_mask):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def __del__(self):
        try:
            self._kill_daemon()
        except Exception:
            pass
