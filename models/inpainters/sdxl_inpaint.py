"""SDXL WebUI 인페인팅 클라이언트.

모든 WebUI 페이로드 파라미터를 __init__ 인자로 노출 — config.yaml 의 inpainter
섹션 키들이 그대로 페이로드 키와 매핑된다. 모델 선택은 per-request
`override_settings.sd_model_checkpoint` 로 명시해서 서버 전역 상태에 의존하지
않는다.

레퍼런스: Notion (sd_xl_inpainting_1.0 워크플로우) + /home/cjy/workspace/sdxl_webui_api.py
"""
import os
import base64
import io
from typing import Any, Optional

import requests
import numpy as np
import cv2
from PIL import Image

from core.base_models import BaseInpaintingModel


class SDXLWebUIInpaintingModel(BaseInpaintingModel):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7860/sdapi/v1/img2img",
        # ── 모델 선택 (per-request override) ───────────────────────────
        sd_model_checkpoint: Optional[str] = None,   # 예: "sd_xl_inpainting_1.0.safetensors [fe1b97fe65]"
        sd_vae: Optional[str] = None,                # 미지정 시 모델 내장 VAE
        # ── 프롬프트 ───────────────────────────────────────────────────
        prompt: str = "background",
        negative_prompt: str = "person, figure, woman, man, character",
        # ── 샘플링 ─────────────────────────────────────────────────────
        sampler_name: str = "DPM++ 2M",
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        # ── 인페인팅 세부 ──────────────────────────────────────────────
        denoising_strength: float = 0.97,
        mask_blur: int = 9,
        inpainting_fill: int = 0,           # 0=fill, 1=original, 2=latent noise, 3=latent nothing
        inpaint_full_res: bool = True,
        inpaint_full_res_padding: int = 208,
        soft_inpainting: bool = True,
        # ── 출력 해상도 (None → 입력 이미지 크기 사용) ─────────────────
        width: Optional[int] = None,
        height: Optional[int] = None,
        # ── 코드 쪽 전처리 ─────────────────────────────────────────────
        mask_dilate_px: int = 10,           # cv2 dilation: kernel = 2*N+1 (현재 10 → 21x21)
    ):
        self.api_url = api_url
        # /sdapi/v1/img2img 경로를 /sdapi/v1/options 등으로 derive 할 수 있게 base 보관
        self.api_base = api_url.rsplit("/sdapi/", 1)[0] if "/sdapi/" in api_url else None

        self.sd_model_checkpoint = sd_model_checkpoint
        self.sd_vae = sd_vae

        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.sampler_name = sampler_name
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed

        self.denoising_strength = denoising_strength
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.soft_inpainting = soft_inpainting

        self.width = width
        self.height = height
        self.mask_dilate_px = mask_dilate_px

    def load_model(self, **kwargs):
        msg = f"[SDXLWebUIInpaint] WebUI API({self.api_url}) — 가중치 로딩 없음."
        if self.sd_model_checkpoint:
            msg += f" Request 시마다 override_settings 로 '{self.sd_model_checkpoint}' 사용."
        print(msg)

    def _encode_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _decode_base64(self, b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def _build_payload(self, encoded_image: str, encoded_mask: str,
                       image: Image.Image) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "init_images": [encoded_image],
            "mask": encoded_mask,
            "width": self.width if self.width is not None else 1024,
            "height": self.height if self.height is not None else 1024,
            "inpaint_full_res": self.inpaint_full_res,
            "inpaint_full_res_padding": self.inpaint_full_res_padding,
            "mask_blur": self.mask_blur,
            "inpainting_fill": self.inpainting_fill,
            "denoising_strength": self.denoising_strength,
            "sampler_name": self.sampler_name,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
        }

        # 모델/VAE per-request 강제 지정 (Notion: sd_xl_inpainting_1.0.safetensors)
        override: dict[str, Any] = {}
        if self.sd_model_checkpoint:
            override["sd_model_checkpoint"] = self.sd_model_checkpoint
        if self.sd_vae:
            override["sd_vae"] = self.sd_vae
        if override:
            payload["override_settings"] = override
            # 다음 요청에서도 override 유지 (서버 globalstate 복원 안 함)
            payload["override_settings_restore_afterwards"] = False

        # Soft Inpainting 확장
        if self.soft_inpainting:
            payload["alwayson_scripts"] = {"soft inpainting": {"args": [True]}}

        return payload

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[SDXLWebUIInpaint] 인페인팅 시작...")
        if self.sd_model_checkpoint:
            print(f"[SDXLWebUIInpaint] checkpoint: {self.sd_model_checkpoint}")

        # 마스크 dilation (커널 크기 = 2N+1)
        k = 2 * self.mask_dilate_px + 1
        mask_arr = np.array(mask.convert("L"))
        
        #---------------추가-------------#
        # 내부 빈 공간 채우기: 객체의 듬성듬성 빈 픽셀이나 구멍을 완전히 메워줌
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_arr, contours, -1, 255, thickness=cv2.FILLED)
        #--------------------------------#
        
        kernel = np.ones((k, k), np.uint8)
        dilated_mask = Image.fromarray(cv2.dilate(mask_arr, kernel, iterations=1))
        
        #----------추가------------------#
        # 디버깅용: 내부 구멍 채우기 및 팽창이 적용된 최종 마스크 이미지 저장
        debug_mask_path = "/home/cjy/workspace/pipeline/result/mask/debug_processed_mask.png"
        os.makedirs(os.path.dirname(debug_mask_path), exist_ok=True)
        dilated_mask.save(debug_mask_path)
        print(f"[SDXLWebUIInpaint] 처리된 마스크 저장 완료: {debug_mask_path}")
        #--------------------------------#

        encoded_image = self._encode_base64(image)
        encoded_mask = self._encode_base64(dilated_mask)
        payload = self._build_payload(encoded_image, encoded_mask, image)

        resp = requests.post(self.api_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"[SDXLWebUIInpaint] API 요청 실패 ({resp.status_code}): {resp.text}")

        result = resp.json()
        out = self._decode_base64(result["images"][0])
        print("[SDXLWebUIInpaint] 인페인팅 완료.")
        return out
