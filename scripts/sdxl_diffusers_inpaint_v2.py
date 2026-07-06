"""SDXL Inpainting 단발성 추론 스크립트 (Only Masked 지원).

호출 측 (pipeline/models/inpainters/sdxl_diffusers_inpaint_v2.py) 가 conda env
(flux_inpaint 등) 을 활성화한 뒤 이 스크립트를 spawn 한다. 입출력은 모두 파일.

CLI:
    python sdxl_diffusers_inpaint_v2.py \
        --image <input.png> --mask <mask.png> --output <out.png> \
        --prompt "background" --negative_prompt "person, figure, ..." \
        --strength 0.97 --guidance_scale 7 --steps 20 --seed -1 \
        --inference_size 1024 [--model diffusers/stable-diffusion-xl-1.0-inpainting-0.1] \
        --inpaint_full_res --padding 208
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

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
    p.add_argument("--inpaint_full_res", action="store_true",
                   help="WebUI 의 'Inpaint area: Only masked' 기능 활성화")
    p.add_argument("--padding", type=int, default=208,
                   help="inpaint_full_res 사용 시 Only masked padding 값")
    return p.parse_args()


def get_crop_region(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    """마스크 영역을 감싸는 Bounding Box를 구하고 padding만큼 확장합니다.
    SDXL 모델 추론 시 찌그러짐을 방지하기 위해 가급적 1:1 비율(정사각형)에 가깝게 보정합니다.
    """
    mask_np = np.array(mask)
    y_indices, x_indices = np.where(mask_np > 0)
    
    # 마스크가 완전히 비어있는 경우 방어 코드 (전체 반환)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0, 0, mask.width, mask.height
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # 1. 패딩 먼저 적용
    x_min = x_min - padding
    y_min = y_min - padding
    x_max = x_max + padding
    y_max = y_max + padding
    
    # 2. 정사각형 비율로 만들기 (가로 세로 중 긴 쪽에 맞춤)
    w = x_max - x_min
    h = y_max - y_min
    size = max(w, h)
    
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    
    new_x_min = cx - size // 2
    new_x_max = new_x_min + size
    new_y_min = cy - size // 2
    new_y_max = new_y_min + size
    
    # 3. 이미지 경계를 벗어나면 화면 안쪽으로 밀어주기 (Shift)
    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_x_max > mask.width:
        new_x_min -= (new_x_max - mask.width)
        new_x_max = mask.width
        if new_x_min < 0:
            new_x_min = 0
            
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_y_max > mask.height:
        new_y_min -= (new_y_max - mask.height)
        new_y_max = mask.height
        if new_y_min < 0:
            new_y_min = 0
            
    return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))


def main() -> int:
    args = parse_args()
    print(f"[SDXL/diffusers-V2] args: {vars(args)}")

    # 입력 로드
    init_image = Image.open(args.image).convert("RGB")
    mask_image = Image.open(args.mask).convert("L")
    original_size = init_image.size  # (W, H)
    print(f"[SDXL/diffusers-V2] input image: {original_size}, mask: {mask_image.size}")

    # mask_blur: WebUI 와 의미 일치 — 마스크 가장자리 페더링
    # 보통 블러를 먹인 뒤 크롭하고 합성해야 자연스럽게 가장자리가 블렌딩됩니다.
    if args.mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=args.mask_blur))

    crop_box = None
    if args.inpaint_full_res:
        # Bounding box + padding 구하기
        crop_box = get_crop_region(mask_image, args.padding)
        print(f"[SDXL/diffusers-V2] inpaint_full_res=True, crop_box: {crop_box}")
        
        # Bounding Box 영역만큼 크롭
        proc_image = init_image.crop(crop_box)
        proc_mask = mask_image.crop(crop_box)
    else:
        proc_image = init_image
        proc_mask = mask_image

    proc_size = proc_image.size
    
    # SDXL 추론을 위해 inference_size 로 리사이즈 하되, "원본 비율을 완벽히 유지" (찌그러짐 방지)
    sz = args.inference_size
    w, h = proc_size
    if w == h:
        new_w, new_h = sz, sz
    elif w > h:
        new_w = sz
        new_h = int(h * (sz / w))
    else:
        new_h = sz
        new_w = int(w * (sz / h))
        
    # diffusers 모델은 가로세로가 8의 배수여야 합니다.
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    print(f"[SDXL/diffusers-V2] resize for inference: {proc_size} -> {(new_w, new_h)}")
    proc_image_resized = proc_image.resize((new_w, new_h), Image.LANCZOS)
    proc_mask_resized = proc_mask.resize((new_w, new_h), Image.LANCZOS)

    # 모델 로드 — HF 캐시에 이미 받아둔 형식
    print(f"[SDXL/diffusers-V2] loading model: {args.model}")
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[SDXL/diffusers-V2] xformers ON")
    except Exception:
        print("[SDXL/diffusers-V2] xformers 미사용")
    pipe = pipe.to(args.device)

    # 시드
    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # 추론
    print("[SDXL/diffusers-V2] inference 시작...")
    result_resized = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=proc_image_resized,
        mask_image=proc_mask_resized,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    print("[SDXL/diffusers-V2] inference 완료")

    # 결과 이미지를 크롭되었던 크기(proc_size)로 복원
    result_cropped = result_resized.resize(proc_size, Image.LANCZOS)

    if args.inpaint_full_res:
        # 원본 이미지에 최종 합성 (Paste using mask as alpha for seamless blending)
        final_image = init_image.copy()
        final_image.paste(result_cropped, crop_box[:2], proc_mask)
    else:
        # 전체 인페인팅이었을 경우 원본 해상도로 그대로 리사이즈
        final_image = result_cropped.resize(original_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    final_image.save(args.output)
    print(f"[SDXL/diffusers-V2] 저장: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
