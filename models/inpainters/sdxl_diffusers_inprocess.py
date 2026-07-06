"""SDXL Inpainting — 로컬 diffusers **in-process** 인페인터 (B2).

기존 코드 무수정 추가본. WebUI(A1111) 서버(localhost:7860)도, 서브프로세스
(flux_inpaint) 도 쓰지 않고, **파이프라인 프로세스 안에서** diffusers 파이프라인을
1회 로드해 직접 호출한다 (promptgen 과 동일한 in-process 패턴).

WebUI 에만 있던 기능들을 diffusers 코드로 이식/근사:
  - 샘플러 'DPM++ 2M (Karras)'  → DPMSolverMultistepScheduler(use_karras_sigmas=True)  [거의 동일]
  - inpainting_fill=0 ('fill')   → A1111 modules/masking.py:fill 포팅 (다중스케일 blur) [거의 동일]
  - Inpaint area: Only masked    → 마스크 bbox 크롭+패딩 후 합성 (sdxl V2 와 동일 로직)
  - soft inpainting              → 경계 페더 알파 합성으로 **근사** (A1111 의 per-step latent
                                    blend 와 비트 일치는 불가 — 가장자리 점진 블렌딩만 재현)

★ in-process 라 호출하는 env(wise) 에 diffusers 필요 (설치됨: 0.30.3 --no-deps).
★ 출력은 WebUI 와 100% 동일하지 않다 (seed RNG·샘플링 루프 수치·soft 근사 차이).
  대신 누설 제어(strength=1.0/NEAREST)·속도(1회 로드 상주)·서버리스 이점을 가진다.

load_model() = 즉시 로드(in-process, 패턴 (a)). predict() 는 가중치 재로딩 없음.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from core.base_models import BaseInpaintingModel


# ── A1111 modules/masking.py:fill 포팅 (inpainting_fill=0 'fill') ──────────────
def _a1111_fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """마스크 영역을 주변 색의 다중스케일 blur 로 채운다 (A1111 'fill' 모드 동일).

    denoise 전에 마스크 안을 주변색으로 미리 메워, strength<1.0 에서 원본(제거대상)이
    그대로 비치는 걸 줄인다. diffusers 순정 파이프라인엔 이 단계가 없어 직접 이식.
    """
    image_mod = Image.new("RGBA", (image.width, image.height))
    image_masked = Image.new("RGBa", (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"),
                       mask=ImageOps.invert(mask.convert("L")))
    image_masked = image_masked.convert("RGBa")
    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    return image_mod.convert("RGB")


def _get_crop_region(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    """마스크 bbox 를 padding 만큼 키우고 정사각형으로 보정해 화면 안으로 민다.

    sdxl_diffusers_inpaint_v2.py 의 get_crop_region 과 동일 (Only masked 재현).
    """
    arr = np.array(mask.convert("L"))
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return 0, 0, mask.width, mask.height
    x0, x1 = int(xs.min()) - padding, int(xs.max()) + padding
    y0, y1 = int(ys.min()) - padding, int(ys.max()) + padding
    size = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    nx0, nx1 = cx - size // 2, cx - size // 2 + size
    ny0, ny1 = cy - size // 2, cy - size // 2 + size
    if nx0 < 0:
        nx1 -= nx0; nx0 = 0
    if nx1 > mask.width:
        nx0 -= (nx1 - mask.width); nx1 = mask.width; nx0 = max(nx0, 0)
    if ny0 < 0:
        ny1 -= ny0; ny0 = 0
    if ny1 > mask.height:
        ny0 -= (ny1 - mask.height); ny1 = mask.height; ny0 = max(ny0, 0)
    return int(nx0), int(ny0), int(nx1), int(ny1)


class SDXLDiffusersInProcessInpaintingModel(BaseInpaintingModel):
    def __init__(
        self,
        model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        prompt: str = "background",
        negative_prompt: str = "person, figure, woman, man, character",
        # ── 샘플링 ─────────────────────────────────────────────────────────
        scheduler: str = "dpmpp_2m_karras",   # 'dpmpp_2m_karras' | 'default'
        steps: int = 20,
        guidance_scale: float = 7.0,
        strength: float = 1.0,                # robust 기본=1.0 (누설차단). WebUI패리티는 0.97
        seed: int = -1,
        inference_size: int = 1024,
        # ── 마스크 전처리 ──────────────────────────────────────────────────
        mask_dilate_px: int = 10,             # cv2 dilation 커널 = 2N+1
        fill_mask_holes: bool = True,         # 내부 구멍 contour fill (WebUI 클라이언트와 동일)
        mask_blur: int = 4,                   # 경계 페더 반경(px). robust 기본 4
        # ── WebUI 기능 토글 ────────────────────────────────────────────────
        inpainting_fill: str = "fill",        # 'fill'(주변색 prefill) | 'original'(미세전처리 X)
        soft_inpainting: bool = True,         # 경계 페더 알파 합성(근사)
        inpaint_full_res: bool = True,        # Only masked (bbox 크롭)
        inpaint_full_res_padding: int = 208,
        # ── 런타임 ─────────────────────────────────────────────────────────
        device: str = "cuda:0",
        dtype: str = "fp16",
    ):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.scheduler = scheduler
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.seed = seed
        self.inference_size = inference_size
        self.mask_dilate_px = mask_dilate_px
        self.fill_mask_holes = fill_mask_holes
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.soft_inpainting = soft_inpainting
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.device = device
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.pipe = None

    # ── 패턴 (a): 즉시 로드 (가중치를 GPU 에 1회 상주) ─────────────────────
    def load_model(self, **kwargs):
        from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
        print(f"[SDXLInProc] 로드 시작: {self.model} (device={self.device}, dtype={self.dtype})")
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.model, torch_dtype=self.dtype, variant="fp16")
        except Exception:
            # fp16 variant 가 없을 때 폴백
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.model, torch_dtype=self.dtype)

        if self.scheduler == "dpmpp_2m_karras":
            # WebUI 'DPM++ 2M Karras' 매핑
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True,
                algorithm_type="dpmsolver++")
            print("[SDXLInProc] scheduler = DPM++ 2M Karras")

        pipe = pipe.to(self.device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[SDXLInProc] xformers ON")
        except Exception:
            pipe.enable_attention_slicing()
            print("[SDXLInProc] xformers 미사용 → attention slicing")
        self.pipe = pipe
        print("[SDXLInProc] 로드 완료 (상주).")

    # ── 마스크 전처리: 구멍채움 + dilation (이진 유지) ─────────────────────
    def _prep_mask(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask.convert("L"))
        if self.fill_mask_holes:
            contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(arr, contours, -1, 255, thickness=cv2.FILLED)
        if self.mask_dilate_px > 0:
            k = 2 * self.mask_dilate_px + 1
            arr = cv2.dilate(arr, np.ones((k, k), np.uint8), iterations=1)
        arr = (arr > 127).astype(np.uint8) * 255   # 재이진화 (그라데이션 방지)
        return Image.fromarray(arr, mode="L")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("[SDXLInProc] load_model() 를 먼저 호출해야 함.")
        print("[SDXLInProc] in-process SDXL 인페인팅 시작...")
        image = image.convert("RGB")
        orig_size = image.size                       # (W, H)
        bin_mask = self._prep_mask(mask)             # 이진 마스크(구멍채움+dilate)

        # 1) Only masked: bbox 크롭
        if self.inpaint_full_res:
            box = _get_crop_region(bin_mask, self.inpaint_full_res_padding)
            proc_img = image.crop(box)
            proc_mask = bin_mask.crop(box)
        else:
            box = None
            proc_img, proc_mask = image, bin_mask
        proc_size = proc_img.size

        # 2) inpainting_fill='fill': denoise 전 마스크영역 주변색 prefill
        if self.inpainting_fill == "fill":
            proc_img = _a1111_fill(proc_img, proc_mask)

        # 3) 추론 해상도로 리사이즈 (비율 유지, 8의 배수)
        sz = self.inference_size
        w, h = proc_size
        if w >= h:
            nw, nh = sz, max(8, int(h * sz / w))
        else:
            nw, nh = max(8, int(w * sz / h)), sz
        nw, nh = (nw // 8) * 8, (nh // 8) * 8
        # pipe 입력: 이미지는 부드럽게, 마스크는 이진 NEAREST (경계 그라데이션 차단)
        img_in = proc_img.resize((nw, nh), Image.LANCZOS)
        mask_in = proc_mask.resize((nw, nh), Image.NEAREST)

        # 4) 추론
        gen = None
        if self.seed >= 0:
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=img_in,
            mask_image=mask_in,
            num_inference_steps=self.steps,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            generator=gen,
        ).images[0]
        result = result.resize(proc_size, Image.LANCZOS)

        # 5) soft inpainting 근사 = 경계 페더 알파 합성
        #    A1111 의 per-step latent blend 는 못 하지만, 결과를 원본 위에 페더된
        #    알파(=blur 마스크)로 얹어 가장자리 급격한 솔기를 완화한다.
        if self.soft_inpainting and self.mask_blur > 0:
            alpha = proc_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))
        else:
            alpha = proc_mask
        comp = Image.composite(result, proc_img, alpha)

        # 6) Only masked 였으면 원본 전체에 되붙이기
        if box is not None:
            final = image.copy()
            final.paste(comp, box[:2], alpha)
            out = final
        else:
            out = comp
        print("[SDXLInProc] 인페인팅 완료.")
        return out.convert("RGB")
