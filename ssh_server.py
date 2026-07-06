"""
SSH 처리 서버 측 워커(worker) — 백본 통합 버전.

공개 서버(server.py)가 SFTP로 INBOX에 떨어뜨린 작업을 감시하고, 백본 파이프라인
(EditingPipeline = WISE-style segmenter + inpainter) 을 통과시켜 결과를 OUTBOX 에
작성한다. OUTBOX 의 <task_id>.json 출현 = 공개 서버가 회수해갈 "완료 신호".

설정은 worker_config.py 에 직접 써넣는다. 백본 경로 / config 도 거기서 지정.

★ 실행 전: 반드시 `conda activate WISE` 후 `python ssh_server.py` 실행할 것.
   wise_seg 가 import 하는 transformers / sam2 / qwen_vl_utils / flash-attn 이
   이 env 안에 설치돼 있다 (NOTES.md A-4 참고).

⚠ 메모 B-3 (직렬 처리는 의도가 아니라 우연):
    main 루프 `for tid in find_pending_tasks(): process_one(tid)` 가 한 번에
    한 task 만 처리해 GPU 경합을 막아준다. 누가 ThreadPoolExecutor 로 바꾸면
    즉시 GPU 충돌. 향후 명시적 lock 필요.

⚠ 메모 (백본 로드 비용):
    _init_pipeline() 는 한 번 호출 시 GPU 모델 weights 를 메모리에 올린다
    (WISE 7B + SAM2 = ~14-15GB VRAM). 워커 부팅 시 1회만 호출하면 이후
    process_one 마다 weights 재로딩 없이 빠르게 추론. 다만 LaMa/SD/MAT 같은
    subprocess inpainter 는 매번 새 conda env 에서 weights 재로딩 — 호출당
    오버헤드 큼. (NOTES.md A-4/B-2 참고)
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from typing import Callable, Optional

import worker_config as cfg

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ssh_server")

# 종료 신호. 현재 작업은 마치고 루프만 빠져나오게 한다.
_stop = False

# OUTBOX 청소 주기 / 보존 기간
_CLEANUP_INTERVAL_SEC = 300
_OUTBOX_STALE_AGE_SEC = 3600
_last_cleanup_ts = 0.0

# 백본 파이프라인 — 워커 lifetime 내내 1회만 생성 (heavy GPU load).
_pipeline = None
_pipeline_lock = threading.Lock()


def _on_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    log.info("shutdown signal=%s 수신, 진행 중 작업 마치고 종료", signum)


# ── 사용자 친화 에러 문자열 (B-11) ──────────────────────────────────────
def safe_error_str(e: BaseException) -> str:
    try:
        return str(e).encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        try:
            return repr(e)
        except Exception:
            return "unknown error"


# ── Atomic 쓰기 헬퍼 ─────────────────────────────────────────────────────
def write_atomic(target_path: str, content: bytes | str) -> None:
    directory = os.path.dirname(target_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            if isinstance(content, str):
                content = content.encode("utf-8")
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_stage(task_id: str, msg: str) -> None:
    write_atomic(f"{cfg.OUTBOX_DIR}/{task_id}.stage", msg)


def write_result(task_id: str, text: str, image_filename: str = "") -> None:
    body = json.dumps({"text": text, "image": image_filename},
                      ensure_ascii=False).encode("utf-8")
    write_atomic(f"{cfg.OUTBOX_DIR}/{task_id}.json", body)


def move_into_outbox(local_path: str, dst_name: str) -> str:
    dst = f"{cfg.OUTBOX_DIR}/{dst_name}"
    try:
        if os.stat(local_path).st_dev == os.stat(cfg.OUTBOX_DIR).st_dev:
            shutil.move(local_path, dst)
            return dst
    except FileNotFoundError:
        pass

    with open(local_path, "rb") as f:
        write_atomic(dst, f.read())
    try:
        os.remove(local_path)
    except OSError:
        pass
    return dst


# ── 작업 큐 / 점유 ───────────────────────────────────────────────────────
def find_pending_tasks() -> list[str]:
    return sorted({
        os.path.basename(p).removesuffix(".json")
        for p in glob.glob(f"{cfg.INBOX_DIR}/*.json")
    })


def claim(task_id: str) -> Optional[tuple[str, str, dict]]:
    src_jpg = f"{cfg.INBOX_DIR}/{task_id}.jpg"
    src_json = f"{cfg.INBOX_DIR}/{task_id}.json"
    if not (os.path.isfile(src_jpg) and os.path.isfile(src_json)):
        return None

    dst_jpg = f"{cfg.WORK_DIR}/{task_id}.jpg"
    dst_json = f"{cfg.WORK_DIR}/{task_id}.json"
    try:
        os.rename(src_json, dst_json)
        os.rename(src_jpg, dst_jpg)
    except FileNotFoundError:
        return None

    with open(dst_json, "r", encoding="utf-8") as f:
        meta_in = json.load(f)
    return dst_jpg, dst_json, meta_in


# ── Cancel 마커 체크 ─────────────────────────────────────────────────────
class _Cancelled(Exception):
    """단말 측 timeout / 사용자 취소 → 공개 서버가 INBOX 에 작성한
    <task_id>.cancel 마커 발견 시 발생."""


def _check_cancel(task_id: str) -> None:
    if os.path.isfile(f"{cfg.INBOX_DIR}/{task_id}.cancel"):
        raise _Cancelled()


def _cleanup_outbox_for_task(task_id: str) -> None:
    patterns = [
        f"{cfg.OUTBOX_DIR}/{task_id}.json",
        f"{cfg.OUTBOX_DIR}/{task_id}.stage",
        f"{cfg.OUTBOX_DIR}/{task_id}_out.jpg",
        f"{cfg.OUTBOX_DIR}/{task_id}_out.*",
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        os.remove(f"{cfg.INBOX_DIR}/{task_id}.cancel")
    except OSError:
        pass


def cleanup_outbox_stale(max_age_sec: int = _OUTBOX_STALE_AGE_SEC) -> int:
    """공개 서버 pull() 회수 실패 등으로 두고 간 OUTBOX 잔여물 청소."""
    now = time.time()
    removed = 0
    for path in glob.glob(f"{cfg.OUTBOX_DIR}/*"):
        try:
            if os.path.isfile(path) and (now - os.stat(path).st_mtime) > max_age_sec:
                os.remove(path)
                log.info("cleanup_outbox_stale: removed %s", path)
                removed += 1
        except OSError:
            pass
    return removed


# ── 백본 파이프라인 초기화 ───────────────────────────────────────────────
def _ensure_pipeline_on_sys_path() -> None:
    if cfg.PIPELINE_ROOT and cfg.PIPELINE_ROOT not in sys.path:
        sys.path.insert(0, cfg.PIPELINE_ROOT)


def init_pipeline():
    """백본 파이프라인을 모든 모델 weights 와 함께 메모리에 올린다.

    워커 부팅 시 1회 호출하면 이후 process_one() 호출마다 빠르게 재사용.
    재진입 안전 (lock + memoize).
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        _ensure_pipeline_on_sys_path()

        # lazy import — sys.path 가 잡힌 뒤에야 백본 모듈을 찾을 수 있다.
        try:
            from main import build_segmenter, build_inpainter, load_config  # type: ignore
            from core.pipeline import EditingPipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"백본 import 실패 — PIPELINE_ROOT={cfg.PIPELINE_ROOT!r} 확인. {e}"
            )

        if cfg.PIPELINE_CONFIG_OVERRIDE is not None:
            pipeline_cfg = cfg.PIPELINE_CONFIG_OVERRIDE
            log.info("pipeline config: PIPELINE_CONFIG_OVERRIDE 사용")
        else:
            pipeline_cfg = load_config(cfg.PIPELINE_CONFIG_PATH)
            log.info("pipeline config: %s 로드", cfg.PIPELINE_CONFIG_PATH)

        seg_name = pipeline_cfg["models"]["segmenter"]["name"]
        inp_name = pipeline_cfg["models"]["inpainter"]["name"]
        log.info("pipeline 모델 인스턴스화: segmenter=%s, inpainter=%s", seg_name, inp_name)

        seg = build_segmenter(pipeline_cfg["models"]["segmenter"])
        inp = build_inpainter(pipeline_cfg["models"]["inpainter"])

        log.info("pipeline weights 로드 중 (heavy 모델이면 수십초 걸릴 수 있음)...")
        t0 = time.time()
        seg.load_model()
        inp.load_model()
        log.info("pipeline weights 로드 완료 (%.1fs)", time.time() - t0)

        _pipeline = EditingPipeline(seg, inp)
        return _pipeline


# ── 단일 task 처리 ───────────────────────────────────────────────────────
def process_one(task_id: str) -> None:
    claimed = claim(task_id)
    if claimed is None:
        return
    image_path, meta_in_path, meta_in = claimed
    prompt = meta_in.get("text", "")
    log.info("claimed task=%s prompt=%r", task_id, prompt)

    image_filename = f"{task_id}_out.jpg"
    work_output = os.path.join(cfg.WORK_DIR, f"{task_id}_out.jpg")

    # 백본 파이프라인의 on_stage 콜백 — 매 단계 직전 cancel 체크 + stage 텍스트 작성.
    def stage_cb(msg: str) -> None:
        _check_cancel(task_id)
        try:
            write_stage(task_id, msg)
        except Exception as e:
            log.warning("write_stage error (continue): %s", e)

    try:
        _check_cancel(task_id)
        pipeline = init_pipeline()

        # 백본은 work_output 에 결과 이미지를 저장. on_stage 로 단계 전파.
        pipeline.run(
            image_path=image_path,
            prompt=prompt,
            output_path=work_output,
            on_stage=stage_cb,
        )

        _check_cancel(task_id)

        # WORK → OUTBOX 로 원자 이동 (같은 파일시스템 가정 시 shutil.move 가 rename).
        move_into_outbox(work_output, image_filename)

        # 마지막에 메타 JSON (= 완료 신호)
        write_result(task_id, "처리 완료", image_filename)
        log.info("done task=%s image=%s", task_id, image_filename)

    except _Cancelled:
        log.info("task=%s cancelled — OUTBOX 잔여물 + WORK 산출물 청소", task_id)
        _cleanup_outbox_for_task(task_id)
        try:
            os.remove(work_output)
        except OSError:
            pass

    except Exception as e:
        log.exception("process failed task=%s: %s", task_id, e)
        err_text = safe_error_str(e)
        try:
            write_stage(task_id, f"오류: {err_text}")
        except Exception:
            pass
        try:
            write_result(task_id, f"처리 중 오류 발생: {err_text}", "")
        except Exception:
            log.exception("write_result on error failed task=%s", task_id)
        try:
            os.remove(work_output)
        except OSError:
            pass

    finally:
        # WORK 의 입력 파일 정리 (jpg/json). OUTBOX 회수는 공개 서버 pull() 책임.
        for p in (image_path, meta_in_path):
            try:
                os.remove(p)
            except OSError:
                pass


# ── 메인 루프 ────────────────────────────────────────────────────────────
def main() -> None:
    global _last_cleanup_ts

    os.makedirs(cfg.INBOX_DIR, exist_ok=True)
    os.makedirs(cfg.OUTBOX_DIR, exist_ok=True)
    os.makedirs(cfg.WORK_DIR, exist_ok=True)
    log.info("worker 시작: INBOX=%s OUTBOX=%s WORK=%s interval=%.2fs",
             cfg.INBOX_DIR, cfg.OUTBOX_DIR, cfg.WORK_DIR, cfg.POLL_INTERVAL_SEC)

    # 부팅 시 백본 미리 올려둔다 — 첫 task 의 latency 줄이고 모델 로딩 실패는 빨리 알림.
    try:
        init_pipeline()
    except Exception as e:
        log.exception("pipeline 초기화 실패 — 워커가 사용 불가 상태: %s", e)
        # 그래도 main 루프는 돌리되, 매 task 마다 init_pipeline() 가 재시도하다 실패할 것.

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    try:
        n = cleanup_outbox_stale()
        if n > 0:
            log.info("부팅 cleanup_outbox_stale: %d 개 정리", n)
    except Exception:
        log.exception("startup cleanup error — 계속 진행")
    _last_cleanup_ts = time.time()

    while not _stop:
        try:
            for tid in find_pending_tasks():
                if _stop:
                    break
                process_one(tid)

            if time.time() - _last_cleanup_ts > _CLEANUP_INTERVAL_SEC:
                try:
                    n = cleanup_outbox_stale()
                    if n > 0:
                        log.info("주기 cleanup_outbox_stale: %d 개 정리", n)
                except Exception:
                    log.exception("periodic cleanup error — 계속 진행")
                _last_cleanup_ts = time.time()

        except Exception:
            log.exception("worker 루프 예외 — 계속 진행")
        if _stop:
            break
        time.sleep(cfg.POLL_INTERVAL_SEC)

    log.info("worker 종료")


if __name__ == "__main__":
    main()
