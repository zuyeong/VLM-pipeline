from __future__ import annotations

import os
from typing import Callable, Optional

from PIL import Image, ImageOps

from core.base_models import (
    BaseSegmentationModel, BaseInpaintingModel, BasePromptGenModel,
)


StageCallback = Callable[[str], None]


class EditingPipeline:
    """원본 + 프롬프트를 받아 (0) 프롬프트 생성 (1) 마스크 생성 (2) 인페인팅 편집 수행.

    0단계(promptgen)는 선택적이다:
      - promptgen_model 지정 시: run() 의 `prompt` 인자를 *유저 의도*로 보고,
        promptgen 이 (의도 + 이미지) → 세그용 프롬프트 로 변환한 뒤 세그멘터에 주입.
      - 미지정 시: 기존과 동일하게 `prompt` 를 세그멘터에 그대로 전달 (2-stage, 하위 호환).

    run() 시그니처 변경 이력:
      - 기본: (image_path, prompt) — 단독 CLI 실행용. result/output/<basename> 에 저장.
      - 확장: (image_path, prompt, output_path=, on_stage=) — 워커가 결과를 OUTBOX 로
        바로 떨어뜨리고 단계 텍스트를 단말까지 전파할 수 있게.

    ┌──────────── 남은 작업 메모 ────────────┐
    │ TODO #11 텍스트-만 응답 경로 (보류):    │
    │   wise_seg 가 bbox 못 찾으면 빈 마스크  │
    │   반환 → inpaint 가 원본을 거의 그대로  │
    │   반환 → 단말에 "처리 완료" + 원본 사본 │
    │   = silent failure. 향후 (text, image) │
    │   튜플 반환 형태로 시그니처 확장 시 해결. │
    └─────────────────────────────────────────┘
    """

    def __init__(self,
                 seg_model: BaseSegmentationModel,
                 inpaint_model: BaseInpaintingModel,
                 promptgen_model: Optional[BasePromptGenModel] = None):
        self.seg_model = seg_model
        self.inpaint_model = inpaint_model
        self.promptgen_model = promptgen_model

        # 백본 단독 실행 시 사용하는 결과 보관소. 워커 통합 시 output_path 가 주어지면
        # 인페인팅 결과는 거기 저장되고 이 디렉토리는 마스크 보관에만 쓰인다.
        self.mask_dir = "/home/cjy/workspace/pipeline/result/mask"
        self.output_dir = "/home/cjy/workspace/pipeline/result/output"

        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self,
            image_path: str,
            prompt: str,
            output_path: Optional[str] = None,
            on_stage: Optional[StageCallback] = None) -> str:
        """
        Args:
            image_path: 원본 이미지의 전체 경로.
            prompt: 리즈닝 세그멘테이션 프롬프트.
            output_path: 결과 이미지를 저장할 절대 경로. None 이면 result/output/<basename>.
                        워커 통합 시 OUTBOX/<task_id>_out.jpg 같은 경로를 직접 지정.
            on_stage: 단계 진입 시 호출될 콜백 (msg: str). 예외를 던지면 run() 이 그 예외를
                     그대로 전파 (워커의 cancel 체크 등에 유용).

        Returns:
            저장된 결과 이미지의 경로.
        """
        notify: StageCallback = on_stage if on_stage is not None else (lambda _msg: None)

        # 원본 파일명 추출 — 마스크 저장과 (output_path 미지정 시) 결과 저장에 사용.
        file_name = os.path.basename(image_path)

        notify("이미지 로드 중...")
        print(f"[Pipeline] 원본 이미지 로드: {image_path}")
        original_image = Image.open(image_path)
        original_image = ImageOps.exif_transpose(original_image).convert("RGB")

        # 0. (선택) 프롬프트 생성 — promptgen 이 있으면 `prompt`(유저 의도) → 세그용 프롬프트
        seg_prompt = prompt
        if self.promptgen_model is not None:
            notify("프롬프트 생성 중...")
            print(f"[Pipeline] 프롬프트 생성 시작 (유저 의도: '{prompt}')")
            seg_prompt = self.promptgen_model.predict(prompt, original_image)
            print(f"[Pipeline] 생성된 세그 프롬프트: '{seg_prompt}'")

        # 1. 세그멘테이션 실행
        notify("객체 분할 중...")
        print(f"[Pipeline] 세그멘테이션 시작 (Prompt: '{seg_prompt}')")
        mask_image = self.seg_model.predict(original_image, seg_prompt)

        # 마스크 저장 (디버깅용 — 단독 실행이든 통합이든 result/mask/ 에 같이 저장)
        mask_save_path = os.path.join(self.mask_dir, file_name)
        mask_image.save(mask_save_path)
        print(f"[Pipeline] 마스크 저장: {mask_save_path}")

        # 1.5 silent-failure 게이트 (TODO #11 해결): 마스크가 비면 인페인팅이 원본을
        #     거의 그대로 반환 → "처리 완료"로 보이는 무음 실패. 명시적으로 알리고
        #     원본을 그대로 결과로 저장한 뒤 종료한다 (제거 대상 미검출).
        if mask_image.convert("L").getbbox() is None:
            notify("⚠ 제거 대상을 찾지 못했습니다 — 원본 유지")
            print("[Pipeline] 빈 마스크 → 인페인팅 스킵, 원본을 결과로 저장")
            final_save_path = output_path if output_path is not None \
                else os.path.join(self.output_dir, file_name)
            parent = os.path.dirname(final_save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            original_image.save(final_save_path)
            return final_save_path

        # 2. 인페인팅 실행
        notify("배경 채우는 중...")
        print("[Pipeline] 인페인팅 시작")
        result_image = self.inpaint_model.predict(original_image, mask_image)

        # 4. 결과 저장
        notify("결과 저장 중...")
        final_save_path = output_path if output_path is not None \
            else os.path.join(self.output_dir, file_name)
        # output_path 의 부모 디렉토리가 없을 수도 있음 — 보장.
        parent = os.path.dirname(final_save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        result_image.save(final_save_path)
        print(f"[Pipeline] 최종 결과 저장: {final_save_path}")

        return final_save_path
