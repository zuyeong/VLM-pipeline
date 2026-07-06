"""SDXL Inpainting (로컬 diffusers, subprocess) 인페인터 V2.

WebUI의 Inpaint area: Only masked 기능을 로컬 환경에서 동일하게 재현
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.base_models import BaseInpaintingModel
from core.subproc import run_bash, unique_tmp_dir


# 호출 스크립트 경로 (pipeline/scripts/sdxl_diffusers_inpaint_v2.py)
DEFAULT_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "sdxl_diffusers_inpaint_v2.py",
)


class SDXLDiffusersInpaintingModelV2(BaseInpaintingModel):
    def __init__(
        self,
        conda_env: str = "flux_inpaint",
        script_path: str = DEFAULT_SCRIPT_PATH,
        model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        prompt: str = "background",
        negative_prompt: str = "person, figure, woman, man, character",
        strength: float = 0.97,         
        guidance_scale: float = 7.0,  
        steps: int = 20,               
        seed: int = -1,
        inference_size: int = 1024,    
        mask_blur: int = 9,           
        mask_dilate_px: int = 10,      
        inpaint_full_res: bool = True,  # WebUI 의 'Inpaint area: Only masked'
        inpaint_full_res_padding: int = 208, # Only masked padding
        device: str = "cuda",
        cuda_visible_devices: Optional[str] = None,  # 예: "0" — GPU pin
    ):
        self.conda_env = conda_env
        self.script_path = script_path
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.inference_size = inference_size
        self.mask_blur = mask_blur
        self.mask_dilate_px = mask_dilate_px
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.device = device
        self.cuda_visible_devices = cuda_visible_devices

    def load_model(self, **kwargs):
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(
                f"[SDXLDiffusersInpaintV2] 추론 스크립트가 없음: {self.script_path}")
        print(f"[SDXLDiffusersInpaintV2] 서브프로세스 래퍼 준비 완료 "
              f"(env={self.conda_env}, model={self.model})")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[SDXLDiffusersInpaintV2] 서브프로세스 (diffusers SDXL V2) 인페인팅 시작...")

        # 마스크 dilation (WebUI 인페인터와 동일 — 커널 2N+1)
        k = 2 * self.mask_dilate_px + 1
        mask_arr = np.array(mask.convert("L"))

        #---------------추가-------------#
        # 내부 빈 공간 채우기: 객체의 듬성듬성 빈 픽셀이나 구멍을 완전히 메워줌
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_arr, contours, -1, 255, thickness=cv2.FILLED)
        #--------------------------------#

        # 마스크 경계 팽창(Dilation)
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
        dilated_mask = Image.fromarray(mask_arr, mode="L")

        #----------추가------------------#
        # 디버깅용: 내부 구멍 채우기 및 팽창이 적용된 최종 마스크 이미지 저장
        debug_mask_path = "/home/cjy/workspace/pipeline/result/mask/debug_processed_mask.png"
        os.makedirs(os.path.dirname(debug_mask_path), exist_ok=True)
        dilated_mask.save(debug_mask_path)
        print(f"[SDXLDiffusersInpaintV2] 처리된 마스크 저장 완료: {debug_mask_path}")
        #--------------------------------#

        with unique_tmp_dir("sdxl_diffusers_v2") as tmp_dir:
            in_image = os.path.join(tmp_dir, "image.png")
            in_mask = os.path.join(tmp_dir, "mask.png")
            out_path = os.path.join(tmp_dir, "out.png")
            image.save(in_image)
            dilated_mask.save(in_mask)

            cuda_prefix = (
                f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices} "
                if self.cuda_visible_devices is not None else ""
            )
            # 따옴표가 들어간 prompt/negative_prompt 를 안전하게 전달하기 위해 shell-escape.
            import shlex
            prompt_q = shlex.quote(self.prompt)
            neg_q = shlex.quote(self.negative_prompt)
            model_q = shlex.quote(self.model)

            bash_cmd = f"""
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate {self.conda_env}
            {cuda_prefix}python {shlex.quote(self.script_path)} \\
                --image {shlex.quote(in_image)} \\
                --mask {shlex.quote(in_mask)} \\
                --output {shlex.quote(out_path)} \\
                --prompt {prompt_q} \\
                --negative_prompt {neg_q} \\
                --strength {self.strength} \\
                --guidance_scale {self.guidance_scale} \\
                --steps {self.steps} \\
                --seed {self.seed} \\
                --inference_size {self.inference_size} \\
                --mask_blur {self.mask_blur} \\
                --model {model_q} \\
                --device {shlex.quote(self.device)} \\
                --padding {self.inpaint_full_res_padding}
            """
            
            # inpaint_full_res 플래그 처리
            if self.inpaint_full_res:
                bash_cmd = bash_cmd.strip() + " \\\n                --inpaint_full_res\n"

            print(f"[SDXLDiffusersInpaintV2] (subproc) 실행 중 (env={self.conda_env})...")
            run_bash(bash_cmd, label="SDXLDiffusersInpaintV2")
            print("[SDXLDiffusersInpaintV2] (subproc) 완료.")

            if not os.path.exists(out_path):
                raise FileNotFoundError(
                    f"[SDXLDiffusersInpaintV2] 결과 이미지 없음: {out_path}")

            result = Image.open(out_path).convert("RGB")
            result.load()  # tmp_dir 삭제 전 메모리 로드

        return result
