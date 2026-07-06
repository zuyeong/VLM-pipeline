# 파이프라인 개선 — 0단계 프롬프트 생성기 통합 (3-stage)

기존 `세그 → 인페인트` 2-stage 앞에 **prompt model(Qwen2.5-VL)** 을 선택적 0단계로 추가했다.

```
user(text 한글 가능, img) ─► [0 promptgen] ─► [1 reason seg(people)] ─► [2 inpaint]
                              Qwen2.5-VL-7B      VR-7B + SAM2             SDXL
                              4bit @ GPU0        bf16 @ GPU1             subproc @ GPU0
```

## 무엇이 바뀌었나 (6개, 모두 하위 호환)

| # | 파일 | 변경 |
|---|---|---|
| 1 | `core/base_models.py` | `BasePromptGenModel` 신설 (`predict(user_text, image)->str`) |
| 2 | `models/promptgen/qwen_vl_promptgen.py` (신규) | Qwen2.5-VL 4bit 프롬프트 생성기. base + 옵션 LoRA 어댑터. 어댑터 없으면 base+few-shot(S0) 폴백 |
| 3 | `core/pipeline.py` | 선택적 0단계 + silent-failure 게이트(빈 마스크 → 명시적 "미검출", 원본 유지) |
| 4 | `main.py` | `PROMPTGEN_REGISTRY` + `build_promptgen` + `main()` 배선 |
| 5 | `config.yaml` | `promptgen` 섹션 + 멀티 GPU 분리(seg `cuda:1`) |
| 6 | `models/segmenters/wise_seg.py` | `device_map="auto"` → 지정 device pin (`cuda:1`) |

**config 구조는 그대로** — registry+builder+name 컨벤션에 형제 카테고리(`promptgen`)만 추가. `models.promptgen` 섹션을 주석처리하면 **기존 2-stage 동작 그대로**(유저가 프롬프트 직접 입력).

## GPU 배치 (RTX 4090 24GB × 2)

| GPU | 모델 | VRAM |
|---|---|---|
| GPU0 | promptgen 4bit + SDXL 서브프로세스 | ~6 + 7 = 13GB |
| GPU1 | VR-7B + SAM2 | ~16.5GB |

전부 상주, swap 불필요. config 의 `promptgen.device: cuda:0`, `segmenter.device: cuda:1`, `inpainter.cuda_visible_devices: "0"` 으로 제어.

---

# 유저측 실행 방안

## 0) 전제 — 실행 env 의존성 확인 (1회)

파이프라인을 띄우는 conda env(=VR-7B/sam2/flash-attn 가 있는 env, 보통 `wise`)에서 main.py 를
실행한다. promptgen 이 **같은 프로세스에 in-process 로** 뜨므로, 그 env 가 promptgen 의존성도
가져야 한다. 아래로 확인:

```bash
conda activate <파이프라인_env>     # VR-7B/sam2/flash-attn 가 있는 env
python -c "import sam2, flash_attn, qwen_vl_utils; print('seg deps OK')"
python -c "import peft, bitsandbytes, transformers; print('promptgen deps OK')"
```

- `promptgen deps` 에서 에러나면 → `pip install peft bitsandbytes` (그 env 에).
- 둘 다 OK 면 준비 끝.

> 베이스 가중치는 이미 `/home/cjy/workspace/prompt_model/base/Qwen2.5-VL-7B-Instruct` 에 있다.
> LoRA 어댑터는 아직 없으므로(학습 전) promptgen 은 **base+few-shot** 으로 동작한다.

## 1) 3-stage 실행 (promptgen 켜짐)

`config.yaml` 의 `promptgen` 섹션이 활성(기본값)인 상태에서:

```bash
cd /home/cjy/workspace/pipeline
python main.py --config config.yaml
```
- `Image Path:` → 편집할 이미지 경로
- `Prompt:` → **유저 의도(한글 가능)**. 예: `중앙 여성만 남기고 가장자리 사람들 지워줘`
  - promptgen 이 이걸 영어 VR 프롬프트로 변환 → 세그 → 인페인트.
- 결과: `result/output/<파일명>` (마스크는 `result/mask/`).

## 2) 2-stage 폴백 (promptgen 끔 — 기존 동작)

`config.yaml` 에서 `promptgen:` 블록 전체를 주석처리하면 끝.
- 이때 `Prompt:` 에는 **세그용 프롬프트를 직접**(영어 권장) 입력.
- 코드/다른 설정 변경 불필요.

## 3) LoRA 어댑터가 생기면 (학습 후)

`config.yaml` promptgen 섹션의 `adapter:` 줄 주석을 풀고 경로 지정:
```yaml
    adapter: "/home/cjy/workspace/prompt_model/adapters/v1"
```
어댑터가 로드되면 few-shot 없이 LoRA 가 패턴을 내재화해 생성한다.

## 4) 켜고 끄는 운영 (온디맨드)

상시 서버가 아니므로 — 필요할 때 `python main.py` 로 띄우면 세션 시작 시 모델 로드(콜드,
수십 초)가 1회 발생하고, 이후 입력마다 추론. 끝나면 프로세스 종료로 GPU 회수.

## 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `Unknown promptgen` | config `promptgen.name` 오타. `qwen_vl_promptgen` 인지 확인 |
| promptgen OOM (GPU0) | SDXL 과 동거 중 초과 → SDXL `cuda_visible_devices` 를 `"1"` 로 옮기거나 promptgen `quantization: 4bit` 확인 |
| seg OOM (GPU1) | 다른 프로세스가 GPU1 점유 → `nvidia-smi` 확인 |
| "⚠ 제거 대상을 찾지 못했습니다" | 생성/입력 프롬프트가 빈 마스크 산출 → 프롬프트 표현 조정 (무음 실패 아님, 정상 알림) |
| `peft/bitsandbytes ImportError` | 위 0) 전제 미충족 → 해당 env 에 설치 |

## 워커(ssh_server) 통합 시

`EditingPipeline(seg, inpaint, promptgen_model=None)` 의 3번째 인자에 `build_promptgen(cfg['models']['promptgen'])` 로 만든 인스턴스를 넘기면 3-stage. 안 넘기면 기존 2-stage. `run(image, prompt, output_path, on_stage)` 시그니처는 불변.
