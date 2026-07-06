import glob
import os
from PIL import Image, ImageFilter
from core.base_models import BaseInpaintingModel
from core.subproc import run_bash, unique_tmp_dir


class LamaInpaintingModel(BaseInpaintingModel):
    def __init__(self, script_dir, conda_env, ckpt_path):
        self.script_dir = script_dir
        self.conda_env = conda_env
        self.ckpt_path = ckpt_path

    def load_model(self, **kwargs):
        # 서브프로세스 패턴 — 가중치는 매 predict() 마다 lama_env 안에서 로드됨.
        # 여기서는 경로 검증만.
        print("[LamaInpaint] 서브프로세스 래퍼 초기화 완료 (가중치는 파이프라인 실행 시 lama_env 환경에서 로드됩니다)")
        if not os.path.exists(self.script_dir):
            raise FileNotFoundError(f"LaMa 스크립트 디렉토리를 찾을 수 없습니다: {self.script_dir}")

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("[LamaInpaint] 서브프로세스(lama_env 환경)를 통해 인페인팅을 시작합니다...")

        original_size = image.size

        # 동시 호출 시 충돌 없는 고유 임시 디렉토리.
        # context manager 가 끝날 때 항상 삭제 — 예외로 빠져나가도 누수 없음.
        with unique_tmp_dir("lama_inpaint") as tmp_dir:
            tmp_in_dir = os.path.join(tmp_dir, "input")
            tmp_out_dir = os.path.join(tmp_dir, "output")
            os.makedirs(tmp_in_dir)
            os.makedirs(tmp_out_dir)

            # LaMa 의 입력 이름 규칙: image_name.png + image_name_mask.png
            img_path = os.path.join(tmp_in_dir, "test.png")
            mask_path = os.path.join(tmp_in_dir, "test_mask.png")

            image.save(img_path)

            # 객체 테두리 잔상(halo) 방지를 위해 마스크 팽창(dilation).
            dilated_mask = mask.convert("L").filter(ImageFilter.MaxFilter(15))
            dilated_mask.save(mask_path)

            bash_command = f"""
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate {self.conda_env}
            cd {self.script_dir}
            export PYTHONPATH=$(pwd)
            CUDA_VISIBLE_DEVICES=0 python3 bin/predict.py \\
                model.path={self.ckpt_path} \\
                indir={tmp_in_dir} \\
                outdir={tmp_out_dir}
            """

            print("[LamaInpaint] (Subprocess) predict.py 실행 중...")
            run_bash(bash_command, label="LamaInpaint")
            print("[LamaInpaint] (Subprocess) 실행 완료.")

            result_files = glob.glob(os.path.join(tmp_out_dir, "*.png"))
            if not result_files:
                raise FileNotFoundError("[LamaInpaint] 결과 이미지를 찾을 수 없습니다.")

            result_image = Image.open(result_files[0]).convert("RGB")
            result_image.load()  # tmp_dir 삭제 전에 메모리에 완전히 로드
            result_image = result_image.resize(original_size, Image.LANCZOS)

        return result_image
