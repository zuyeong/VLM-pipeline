"""SDXL Inpainting 단발성 추론 스크립트 (subprocess 호출용).

호출 측 (pipeline/models/inpainters/sdxl_diffusers_inpaint.py) 가 conda env
(flux_inpaint 등) 을 활성화한 뒤 이 스크립트를 spawn 한다. 입출력은 모두 파일.

CLI:
    python sdxl_diffusers_inpaint.py \
        --image <input.png> --mask <mask.png> --output <out.png> \
        --prompt "background" --negative_prompt "person, figure, ..." \
        --strength 0.97 --guidance_scale 7 --steps 20 --seed -1 \
        --inference_size 1024 [--model diffusers/stable-diffusion-xl-1.0-inpainting-0.1]

Notion 워크플로우 (sd_xl_inpainting_1.0 + denoise 0.97 + CFG 7 + Soft Inpainting)
를 diffusers 로 옮긴 형태. inpaint_full_res / mask_blur / soft_inpainting 같은
WebUI 전용 옵션은 직접적 대응이 없어 다음으로 대체:
    - inpaint_full_res=True (Only masked) → 1024x1024 로 리사이즈 추론 후 원본 복원
    - mask_blur=9              → 마스크 dilation 후 PIL Gaussian blur 로 근사
    - soft_inpainting=True     → 위 mask_blur 가 같은 역할 (마스크 가장자리 페더)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForInpainting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="background")
    p.add_argument("--negative_prompt", default="person, figure, woman, man, character")
    p.add_argument("--strength", type=float, default=0.97,
                   help="Notion denoising_strength=0.97 (diffusers 의 strength)")
    p.add_argument("--guidance_scale", type=float, default=7.0,
                   help="Notion CFG Scale=7")
    p.add_argument("--steps", type=int, default=20,
                   help="Notion 미명시 → WebUI 기본 20 유지")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--inference_size", type=int, default=1024,
                   help="SDXL native 해상도. 1024 권장")
    p.add_argument("--mask_blur", type=int, default=9,
                   help="WebUI mask_blur 와 동일 의미 — PIL Gaussian blur 픽셀")
    p.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                   help="HF 캐시에 이미 다운로드된 다이퓨저스 형식 모델")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[SDXL/diffusers] args: {vars(args)}")

    # 입력 로드
    init_image = Image.open(args.image).convert("RGB")
    mask_image = Image.open(args.mask).convert("L")
    original_size = init_image.size  # (W, H)
    print(f"[SDXL/diffusers] input image: {original_size}, mask: {mask_image.size}")

    # SDXL native 1024 로 리사이즈 (Notion 의 "Only masked" 자체는 못 살리지만,
    # 적어도 모델 친화 해상도에서 처리)
    sz = args.inference_size
    init_image = init_image.resize((sz, sz), Image.LANCZOS)
    mask_image = mask_image.resize((sz, sz), Image.LANCZOS)

    # mask_blur: WebUI 와 의미 일치 — 마스크 가장자리 페더링
    if args.mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=args.mask_blur))

    # 모델 로드 — HF 캐시에 이미 받아둔 형식
    print(f"[SDXL/diffusers] loading model: {args.model}")
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[SDXL/diffusers] xformers ON")
    except Exception:
        print("[SDXL/diffusers] xformers 미사용")
    pipe = pipe.to(args.device)

    # 시드
    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # 추론
    print("[SDXL/diffusers] inference 시작...")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    print("[SDXL/diffusers] inference 완료")

    # 원본 해상도 복원
    result = result.resize(original_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    result.save(args.output)
    print(f"[SDXL/diffusers] 저장: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
