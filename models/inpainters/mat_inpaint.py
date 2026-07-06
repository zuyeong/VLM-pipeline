import os
from PIL import Image
from core.base_models import BaseInpaintingModel
from core.subproc import run_bash, unique_tmp_dir


class MatInpaintingModel(BaseInpaintingModel):
    def __init__(self, script_dir, conda_env, ckpt_path):
        self.script_dir = script_dir
        self.conda_env = conda_env
        self.ckpt_path = ckpt_path

    def load_model(self, **kwargs):
        # 서브프로세스 패턴 — 가중치는 매 predict() 마다 mat_env 안에서 로드됨.
        # 여기서는 경로 검증만.
        print("[MatInpaint] 서브프로세스 래퍼 초기화 완료 (가중치는 파이프라인 실행 시 mat_env 환경에서 로드됩니다)")
        if not os.path.exists(self.script_dir):
            raise FileNotFoundError(f"MAT 스크립트 디렉토리를 찾을 수 없습니다: {self.script_dir}")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[MatInpaint] 서브프로세스(mat_env 환경)를 통해 인페인팅을 시작합니다...")

        original_size = image.size

        # 동시 호출 시 충돌 없는 고유 임시 디렉토리 (context manager 가 정리 보장).
        with unique_tmp_dir("mat_inpaint") as tmp_dir:
            img_path = os.path.join(tmp_dir, "input.png")
            mask_path = os.path.join(tmp_dir, "mask.png")
            out_path = os.path.join(tmp_dir, "output.png")

            image.save(img_path)
            mask.save(mask_path)

            # MAT 디렉토리 안에 있는 mat_inference.py 스크립트 경로.
            inference_script = os.path.join(self.script_dir, "mat_inference.py")

            bash_command = f"""
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate {self.conda_env}
            CUDA_VISIBLE_DEVICES=0 python {inference_script} \\
                --img {img_path} \\
                --mask {mask_path} \\
                --out {out_path} \\
                --mat_dir {self.script_dir} \\
                --ckpt {self.ckpt_path}
            """

            print("[MatInpaint] (Subprocess) mat_inference.py 실행 중...")
            run_bash(bash_command, label="MatInpaint")
            print("[MatInpaint] (Subprocess) 실행 완료.")

            if not os.path.exists(out_path):
                raise FileNotFoundError("[MatInpaint] 결과 이미지를 찾을 수 없습니다.")

            result_image = Image.open(out_path).convert("RGB")
            result_image.load()  # tmp_dir 삭제 전에 메모리에 완전히 로드
            result_image = result_image.resize(original_size, Image.LANCZOS)

        return result_image
