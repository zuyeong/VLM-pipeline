"""
이미지 편집 파이프라인 엔트리.

CLI 사용:
    python main.py [--config config.yaml]
    → stdin 으로 image_path / prompt 를 받아 1회 실행.

라이브러리 사용 (예: SSH 워커가 직접 호출):
    from main import load_config, build_segmenter, build_inpainter
    cfg = load_config("config.yaml")
    seg = build_segmenter(cfg['models']['segmenter'])
    inp = build_inpainter(cfg['models']['inpainter'])
    seg.load_model()
    inp.load_model()
    pipeline = EditingPipeline(seg, inp)
    pipeline.run(image_path, prompt)
"""
# 처리 서버 시스템 python 이 3.8 일 수 있어 PEP 585 (`dict[...]`) 같은 3.9+ 문법
# 호환성을 위해 어노테이션을 string 으로 lazy 평가시킨다.
from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import yaml
from PIL import Image, ImageDraw

# 파이프라인 구성 요소
from core.base_models import (
    BaseInpaintingModel, BaseSegmentationModel, BasePromptGenModel,
)
from core.pipeline import EditingPipeline

# NOTE: 각 모델 구현체는 빌더 함수 안에서 lazy import.
# 모델마다 요구하는 패키지 버전이 달라서 모듈 상단에서 한꺼번에 import 하면
# 사용하지 않는 모델 때문에 환경 충돌이 난다.


# ───────────────────────────────────────────────────────────────────────────
# 빌더 함수 — config dict → 모델 인스턴스
# ───────────────────────────────────────────────────────────────────────────

def _build_dummy_seg(cfg: dict) -> BaseSegmentationModel:
    from models.segmenters.dummy_seg import DummySegmentationModel
    return DummySegmentationModel()


def _build_wise_seg(cfg: dict) -> BaseSegmentationModel:
    # WISE 코드 + VR-7B 가중치 (다중 객체 in-process). 단일 객체 Seg-Zero-7B 도
    # reasoning_model_path 만 'Ricky06662/Seg-Zero-7B' 로 바꾸면 그대로 동작.
    from models.segmenters.wise_seg import WiseSegmentationModel, DEFAULT_REASONING_MODEL_PATH
    return WiseSegmentationModel(
        reasoning_model_path=cfg.get('reasoning_model_path', DEFAULT_REASONING_MODEL_PATH),
        segmentation_model_path=cfg.get('segmentation_model_path', 'facebook/sam2-hiera-large'),
        device=cfg.get('device', 'cuda'),   # 멀티 GPU 배치: 예) 'cuda:1'
    )


def _build_glamm_seg(cfg: dict) -> BaseSegmentationModel:
    from models.segmenters.glamm_seg import GlammSegmentationModel
    return GlammSegmentationModel(
        version=cfg.get('version', '/home/cjy/workspace/GLaMM-FullScope'),
    )


def _build_read_seg(cfg: dict) -> BaseSegmentationModel:
    from models.segmenters.read_seg import ReadSegmentationModel
    return ReadSegmentationModel(
        pretrained_model_path=cfg.get('pretrained_model_path', 'rui-qian/READ-LLaVA-v1.5-13B-for-ReasonSeg-testset'),
        vision_tower=cfg.get('vision_tower_path', 'openai/clip-vit-large-patch14-336'),
    )


def _build_instruct_seg(cfg: dict) -> BaseSegmentationModel:
    from models.segmenters.instruct_seg import InstructSegModel
    return InstructSegModel(model_path=cfg.get('model_path', 'model/InstructSeg'))


def _build_pixellm_seg(cfg: dict) -> BaseSegmentationModel:
    from models.segmenters.pixellm_seg import PixelLMSegmentationModel
    return PixelLMSegmentationModel(version=cfg.get('version', './PixelLM-13B/hf_model'))


def _build_vision_reasoner_seg(cfg: dict) -> BaseSegmentationModel:
    """VisionReasoner segmenter — 별도 conda env (visionreasoner_test) 의
    데몬 프로세스를 spawn 해서 JSON IPC 로 통신."""
    import os as _os
    from models.segmenters.vision_reasoner_seg import VisionReasonerSegmentationModel
    default_daemon = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        'scripts', 'vr_daemon.py',
    )
    return VisionReasonerSegmentationModel(
        conda_env_python=cfg.get(
            'conda_env_python',
            '/home/cjy/miniconda3/envs/visionreasoner_test/bin/python',
        ),
        vr_root=cfg.get('vr_root', '/home/cjy/workspace/WISE/VisionReasoner'),
        daemon_script_path=cfg.get('daemon_script_path', default_daemon),
        model_path=cfg.get('model_path', 'pretrained_models/VisionReasoner-7B'),
        task_router_path=cfg.get('task_router_path', 'pretrained_models/TaskRouter-1.5B'),
        segmentation_model_path=cfg.get('segmentation_model_path', 'facebook/sam2-hiera-large'),
        device=cfg.get('device', 'cuda:0'),
        force_task=cfg.get('force_task', 'segmentation'),
        startup_timeout_sec=cfg.get('startup_timeout_sec', 600),
        predict_timeout_sec=cfg.get('predict_timeout_sec', 180),
        daemon_log_path=cfg.get('daemon_log_path'),
    )


def _build_dummy_inpaint(cfg: dict) -> BaseInpaintingModel:
    from models.inpainters.dummy_inpaint import DummyInpaintingModel
    return DummyInpaintingModel()


def _build_sd_inpaint(cfg: dict) -> BaseInpaintingModel:
    from models.inpainters.sd_inpaint import SDInpaintingModel
    return SDInpaintingModel(
        script_dir=cfg.get('script_dir', '/home/cjy/workspace/Stable-Diffusion-Inpaint'),
        conda_env=cfg.get('conda_env', 'ldm'),
        ckpt_path=cfg.get('ckpt_path', 'models/ldm/inpainting_big/last.ckpt'),
        yaml_profile=cfg.get('yaml_profile', 'models/ldm/inpainting_big/config.yaml'),
    )


def _build_lama_inpaint(cfg: dict) -> BaseInpaintingModel:
    from models.inpainters.lama_inpaint import LamaInpaintingModel
    return LamaInpaintingModel(
        script_dir=cfg.get('script_dir', '/home/cjy/workspace/lama/lama_repo'),
        conda_env=cfg.get('conda_env', 'lama_env'),
        ckpt_path=cfg.get('ckpt_path', '/home/cjy/workspace/lama/big-lama'),
    )


def _build_mat_inpaint(cfg: dict) -> BaseInpaintingModel:
    from models.inpainters.mat_inpaint import MatInpaintingModel
    return MatInpaintingModel(
        script_dir=cfg.get('script_dir', '/home/cjy/workspace/MAT'),
        conda_env=cfg.get('conda_env', 'mat_env'),
        ckpt_path=cfg.get('ckpt_path', '/home/cjy/workspace/MAT/checkpoints/Places_512_FullData_G.pth'),
    )


def _build_sdxl_inpaint(cfg: dict) -> BaseInpaintingModel:
    # config.yaml 의 inpainter 섹션 키들이 SDXLWebUIInpaintingModel.__init__ 인자명과
    # 그대로 일치하도록 매핑. 'name' 키만 빌더용이라 제거하고 나머지는 통째로 전달.
    from models.inpainters.sdxl_inpaint import SDXLWebUIInpaintingModel
    kwargs = {k: v for k, v in cfg.items() if k != 'name'}
    return SDXLWebUIInpaintingModel(**kwargs)


def _build_sdxl_diffusers_inpaint(cfg: dict) -> BaseInpaintingModel:
    # SDXL 로컬 (diffusers, subprocess). WebUI 서버 불필요.
    from models.inpainters.sdxl_diffusers_inpaint import SDXLDiffusersInpaintingModel
    kwargs = {k: v for k, v in cfg.items() if k != 'name'}
    return SDXLDiffusersInpaintingModel(**kwargs)


def _build_sdxl_diffusers_inpaint_v2(cfg: dict) -> BaseInpaintingModel:
    # SDXL 로컬 (diffusers, subprocess) V2 - Inpaint Only Masked 지원
    from models.inpainters.sdxl_diffusers_inpaint_v2 import SDXLDiffusersInpaintingModelV2
    kwargs = {k: v for k, v in cfg.items() if k != 'name'}
    return SDXLDiffusersInpaintingModelV2(**kwargs)


def _build_sdxl_diffusers_inprocess(cfg: dict) -> BaseInpaintingModel:
    # SDXL 로컬 (diffusers, in-process). 서버/서브프로세스 불필요. WebUI 기능 이식/근사.
    from models.inpainters.sdxl_diffusers_inprocess import SDXLDiffusersInProcessInpaintingModel
    kwargs = {k: v for k, v in cfg.items() if k != 'name'}
    return SDXLDiffusersInProcessInpaintingModel(**kwargs)


def _build_qwen_vl_promptgen(cfg: dict) -> BasePromptGenModel:
    # Qwen2.5-VL 프롬프트 생성기 (in-process, 4bit). base + 옵션 LoRA 어댑터.
    from models.promptgen.qwen_vl_promptgen import QwenVLPromptGenModel
    kwargs = {k: v for k, v in cfg.items() if k != 'name'}
    return QwenVLPromptGenModel(**kwargs)


# 레지스트리 — 새 모델 추가 시 여기 한 줄 + 위에 _build_xxx 함수 1개만 추가.
SEGMENTER_REGISTRY: dict[str, Callable[[dict], BaseSegmentationModel]] = {
    'dummy_seg':    _build_dummy_seg,
    'wise_seg':     _build_wise_seg,
    'glamm_seg':    _build_glamm_seg,
    'read_seg':     _build_read_seg,
    'instruct_seg': _build_instruct_seg,
    'pixellm_seg':  _build_pixellm_seg,
    'vision_reasoner_seg': _build_vision_reasoner_seg,
}

INPAINTER_REGISTRY: dict[str, Callable[[dict], BaseInpaintingModel]] = {
    'dummy_inpaint':           _build_dummy_inpaint,
    'sd_inpaint':              _build_sd_inpaint,
    'lama_inpaint':            _build_lama_inpaint,
    'mat_inpaint':             _build_mat_inpaint,
    'sdxl_inpaint':            _build_sdxl_inpaint,             # WebUI HTTP 클라이언트
    'sdxl_diffusers_inpaint':  _build_sdxl_diffusers_inpaint,   # 로컬 diffusers (subprocess)
    'sdxl_diffusers_inpaint_v2': _build_sdxl_diffusers_inpaint_v2, # 로컬 diffusers V2 (Only Masked)
    'sdxl_diffusers_inprocess': _build_sdxl_diffusers_inprocess,   # 로컬 diffusers in-process (B2, 서버리스)
}

# 0단계 프롬프트 생성기 — 선택적. config 에 models.promptgen 섹션이 있을 때만 사용.
PROMPTGEN_REGISTRY: dict[str, Callable[[dict], BasePromptGenModel]] = {
    'qwen_vl_promptgen': _build_qwen_vl_promptgen,
}


def build_segmenter(seg_cfg: dict) -> BaseSegmentationModel:
    """config['models']['segmenter'] dict → 인스턴스화된 세그멘테이션 모델."""
    name = seg_cfg.get('name')
    if name not in SEGMENTER_REGISTRY:
        raise ValueError(
            f"Unknown segmenter: {name!r}. "
            f"Available: {sorted(SEGMENTER_REGISTRY.keys())}"
        )
    return SEGMENTER_REGISTRY[name](seg_cfg)


def build_inpainter(inp_cfg: dict) -> BaseInpaintingModel:
    """config['models']['inpainter'] dict → 인스턴스화된 인페인팅 모델."""
    name = inp_cfg.get('name')
    if name not in INPAINTER_REGISTRY:
        raise ValueError(
            f"Unknown inpainter: {name!r}. "
            f"Available: {sorted(INPAINTER_REGISTRY.keys())}"
        )
    return INPAINTER_REGISTRY[name](inp_cfg)


def build_promptgen(pg_cfg: dict) -> BasePromptGenModel:
    """config['models']['promptgen'] dict → 인스턴스화된 프롬프트 생성 모델."""
    name = pg_cfg.get('name')
    if name not in PROMPTGEN_REGISTRY:
        raise ValueError(
            f"Unknown promptgen: {name!r}. "
            f"Available: {sorted(PROMPTGEN_REGISTRY.keys())}"
        )
    return PROMPTGEN_REGISTRY[name](pg_cfg)


# ───────────────────────────────────────────────────────────────────────────
# CLI 보조 함수
# ───────────────────────────────────────────────────────────────────────────

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_test_image(path):
    """테스트용 이미지가 없을 때 임시로 생성"""
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((200, 250), "Test Image", fill=(255, 255, 0))
    img.save(path)
    print(f"테스트용 이미지를 생성했습니다: {path}")


def _read_input_path() -> str:
    image_path = input("Image Path: ").strip()
    # 터미널 드래그 앤 드롭 시 묻어오는 따옴표 제거
    if image_path.startswith("'") and image_path.endswith("'"):
        image_path = image_path[1:-1]
    elif image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    return image_path


# ───────────────────────────────────────────────────────────────────────────
# CLI 엔트리
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Editing Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    print("\n----------------------")
    image_path = _read_input_path()
    prompt = input("Prompt: ").strip()
    print("----------------------\n")

    start_time = time.time()

    # 1. 테스트용 이미지가 없다면 임시 생성
    if not os.path.exists(image_path):
        create_test_image(image_path)

    # 2. config 기반 모델 인스턴스화 (워커도 같은 빌더들을 사용)
    models_cfg = config['models']
    seg_model = build_segmenter(models_cfg['segmenter'])
    inpaint_model = build_inpainter(models_cfg['inpainter'])
    # promptgen 섹션은 선택적 — 있으면 0단계 삽입, 없으면 기존 2-stage (하위 호환).
    promptgen_model = None
    if models_cfg.get('promptgen'):
        promptgen_model = build_promptgen(models_cfg['promptgen'])

    # 3. 모델 가중치 로드
    seg_model.load_model()
    inpaint_model.load_model()
    if promptgen_model is not None:
        promptgen_model.load_model()

    # 4. 파이프라인 구성 및 실행
    pipeline = EditingPipeline(seg_model, inpaint_model, promptgen_model)
    pipeline.run(image_path, prompt)

    elapsed_time = time.time() - start_time
    print(f"\n최종 출력까지 걸린 시간: {elapsed_time:.2f}초")


if __name__ == "__main__":
    main()
