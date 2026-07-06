"""
파이프라인 모델들이 따르는 두 가지 추상 베이스.

설계 메모 — load_model() 의 의미:
  같은 메서드명 아래 두 가지 동작 패턴이 공존한다. 호출자는 어느 패턴이든
  "load_model() 1회 → predict() 반복" 의 사용 흐름을 가정해도 안전하다.

  (a) 즉시 로드 (in-process 모델)
      - 예: WiseSegmentationModel — Qwen2.5-VL + SAM2 를 메모리/GPU 에 올림
      - load_model() 비용 큼, 이후 predict() 는 weights 재로딩 없음

  (b) 지연 로드 (외부 conda 환경 + 서브프로세스 모델)
      - 예: LamaInpaintingModel, SDInpaintingModel, MatInpaintingModel
      - load_model() 은 경로/환경 검증만, weights 는 매 predict() 마다
        서브프로세스 안에서 새로 로드 (의존성 격리를 위해 별도 conda env 사용)
      - 호출당 오버헤드가 큼 — 라운드트립 latency 의 주된 원인

신규 모델을 추가할 때는 위 두 패턴 중 어느 쪽인지 클래스 docstring 에 명시 권장.
"""
from abc import ABC, abstractmethod
from PIL import Image


class BaseSegmentationModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        """모델 사용 전 1회 준비.

        구현체에 따라:
          - in-process 모델: 가중치를 메모리/GPU 에 즉시 로드.
          - 서브프로세스 모델: 경로/환경 검증만 수행 (실제 로드는 predict() 내부에서).

        어느 경우든 호출자는 load_model() 을 1회 호출한 뒤 predict() 를 반복 호출.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Input: 원본 이미지(PIL Image), 프롬프트(str)
        Output: 마스크 이미지(PIL Image, "L" 흑백 모드)
        """
        pass


class BaseInpaintingModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        """모델 사용 전 1회 준비. 의미는 BaseSegmentationModel.load_model 참고."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Input: 원본 이미지(PIL Image), 마스크 이미지(PIL Image)
        Output: 인페인팅이 완료된 결과 이미지(PIL Image, "RGB")
        """
        pass


class BasePromptGenModel(ABC):
    """user 의도(+이미지) → 하류 세그멘터용 프롬프트 문자열을 생성하는 0단계 모델.

    파이프라인에서 segmenter 앞에 선택적으로 끼는 stage. 없으면(=config 에
    promptgen 섹션 미지정) 파이프라인은 기존 2-stage 로 동작(하위 호환).
    """

    @abstractmethod
    def load_model(self, **kwargs):
        """모델 사용 전 1회 준비. 의미는 BaseSegmentationModel.load_model 참고."""
        pass

    @abstractmethod
    def predict(self, user_text: str, image: Image.Image) -> str:
        """
        Input: 유저 의도 텍스트(한글 가능), 원본 이미지(PIL Image)
        Output: 하류 세그멘터에 넣을 프롬프트(str). 영어 권장.
        """
        pass
