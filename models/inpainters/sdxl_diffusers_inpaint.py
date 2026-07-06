"""SDXL Inpainting (로컬 diffusers, subprocess) 인페인터.

WebUI 클라이언트(sdxl_inpaint.py) 와 같은 Notion 파라미터를 받지만, 외부 서버
대신 같은 머신의 flux_inpaint 등 conda env 에서 diffusers AutoPipelineForInpainting
을 spawn 하여 추론. LaMa/MAT/SD 인페인터와 동일한 subprocess 패턴.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.base_models import BaseInpaintingModel
from core.subproc import run_bash, unique_tmp_dir


# 기본 호출 스크립트 경로 (pipeline/scripts/sdxl_diffusers_inpaint.py)
DEFAULT_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "sdxl_diffusers_inpaint.py",
)


class SDXLDiffusersInpaintingModel(BaseInpaintingModel):
    def __init__(
        self,
        conda_env: str = "flux_inpaint",
        script_path: str = DEFAULT_SCRIPT_PATH,
        model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        # Notion 워크플로우 파라미터
        prompt: str = "background",
        negative_prompt: str = "person, figure, woman, man, character",
        strength: float = 0.97,         # Notion denoising_strength
        guidance_scale: float = 7.0,    # Notion CFG Scale
        steps: int = 20,                # Notion 미명시 → WebUI 기본 유지
        seed: int = -1,
        inference_size: int = 1024,     # SDXL native
        mask_blur: int = 9,             # Notion: 9 (Soft Inpainting 의 페더링 효과 근사)
        mask_dilate_px: int = 10,       # WebUI 와 동일한 마스크 dilation
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
        self.device = device
        self.cuda_visible_devices = cuda_visible_devices

    def load_model(self, **kwargs):
        # 가중치는 매 predict() 마다 subprocess 안에서 로드 → 여기서는 경로만 검증.
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(
                f"[SDXLDiffusersInpaint] 추론 스크립트가 없음: {self.script_path}")
        print(f"[SDXLDiffusersInpaint] 서브프로세스 래퍼 준비 완료 "
              f"(env={self.conda_env}, model={self.model})")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[SDXLDiffusersInpaint] 서브프로세스 (diffusers SDXL) 인페인팅 시작...")

        # 마스크 dilation (WebUI 인페인터와 동일 — 커널 2N+1)
        k = 2 * self.mask_dilate_px + 1
        mask_arr = np.array(mask.convert("L"))
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
        dilated_mask = Image.fromarray(mask_arr, mode="L")

        with unique_tmp_dir("sdxl_diffusers") as tmp_dir:
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
                --device {shlex.quote(self.device)}
            """

            print(f"[SDXLDiffusersInpaint] (subproc) 실행 중 (env={self.conda_env})...")
            run_bash(bash_cmd, label="SDXLDiffusersInpaint")
            print("[SDXLDiffusersInpaint] (subproc) 완료.")

            if not os.path.exists(out_path):
                raise FileNotFoundError(
                    f"[SDXLDiffusersInpaint] 결과 이미지 없음: {out_path}")

            result = Image.open(out_path).convert("RGB")
            result.load()  # tmp_dir 삭제 전 메모리 로드

        return result
