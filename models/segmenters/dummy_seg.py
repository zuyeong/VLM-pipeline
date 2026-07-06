import time
from PIL import Image, ImageDraw
from core.base_models import BaseSegmentationModel

class DummySegmentationModel(BaseSegmentationModel):
    def load_model(self, **kwargs):
        print("[DummySeg] 모델 가중치를 로드합니다... (가짜 로드)")
        time.sleep(0.5)

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        print(f"[DummySeg] '{prompt}' 프롬프트에 대한 마스크를 생성합니다...")
        time.sleep(1)
        
        # 테스트를 위해 원본 이미지와 같은 크기의 빈 검은색 이미지를 만들고, 가운데에 흰색 원(마스크)을 그림
        mask = Image.new("L", image.size, 0) # "L" 모드: 흑백(8-bit pixels, black and white)
        draw = ImageDraw.Draw(mask)
        
        width, height = image.size
        # 중앙에 원형 마스크 생성
        box = (width//4, height//4, width*3//4, height*3//4)
        draw.ellipse(box, fill=255)
        
        return mask
