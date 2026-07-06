import glob
import os
from PIL import Image
from core.base_models import BaseInpaintingModel
from core.subproc import run_bash, unique_tmp_dir


class SDInpaintingModel(BaseInpaintingModel):
    def __init__(self, script_dir, conda_env, ckpt_path, yaml_profile):
        self.script_dir = script_dir
        self.conda_env = conda_env
        self.ckpt_path = ckpt_path
        self.yaml_profile = yaml_profile

    def load_model(self, **kwargs):
        # 서브프로세스 패턴 — 가중치는 매 predict() 마다 ldm 환경에서 로드됨.
        # 여기서는 경로 검증만.
        print("[SDInpaint] 서브프로세스 래퍼 초기화 완료 (가중치는 파이프라인 실행 시 ldm 환경에서 로드됩니다)")
        if not os.path.exists(self.script_dir):
            raise FileNotFoundError(f"SD Inpaint 스크립트 디렉토리를 찾을 수 없습니다: {self.script_dir}")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[SDInpaint] 서브프로세스(ldm 환경)를 통해 인페인팅을 시작합니다...")

        original_size = image.size

        # 동시 호출 시 충돌 없는 고유 임시 디렉토리 (context manager 가 정리 보장).
        with unique_tmp_dir("sd_inpaint") as tmp_dir:
            tmp_out_dir = os.path.join(tmp_dir, "out")
            os.makedirs(tmp_out_dir)

            # SD Inpaint 입력 이름 규칙: input.png + input_mask.png
            img_path = os.path.join(tmp_dir, "input.png")
            mask_path = os.path.join(tmp_dir, "input_mask.png")

            image.save(img_path)
            mask.save(mask_path)

            bash_command = f"""
            # conda 초기화 스크립트 소싱 (bash 쉘에서 conda 명령어 사용 가능하게)
            source ~/miniconda3/etc/profile.d/conda.sh

            # ldm 환경 활성화
            conda activate {self.conda_env}

            # 스크립트 디렉토리로 이동
            cd {self.script_dir}

            # SD Inpaint 추론 스크립트 실행
            CUDA_VISIBLE_DEVICES=0 python inpaint_inference.py \\
                --indir {tmp_dir} \\
                --outdir {tmp_out_dir} \\
                --prefix test \\
                --ckpt {self.ckpt_path} \\
                --yaml_profile {self.yaml_profile} \\
                --device cuda
            """

            print("[SDInpaint] (Subprocess) inpaint_inference.py 실행 중...")
            run_bash(bash_command, label="SDInpaint")
            print("[SDInpaint] (Subprocess) 실행 완료.")

            # SD Inpaint 는 결과 파일명이 복잡 — glob 으로 첫 png 찾기.
            result_files = glob.glob(os.path.join(tmp_out_dir, "*.png"))
            if not result_files:
                raise FileNotFoundError("[SDInpaint] 결과 이미지를 찾을 수 없습니다.")

            result_image = Image.open(result_files[0]).convert("RGB")
            result_image.load()  # tmp_dir 삭제 전에 메모리에 완전히 로드
            result_image = result_image.resize(original_size, Image.LANCZOS)

        return result_image
