import cv2
import numpy as np
import os
import re
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────────
INPUT_DIR  = "/home/cjy/workspace/SSISv2/output/combined_mask"
OUTPUT_DIR = "/home/cjy/workspace/SSISv2/output/combined_mask_5px"
DILATE_PX  = 5   # 팽창 픽셀 수
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

groups = defaultdict(list)

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.lower().endswith(".png"):
        continue
    match = re.match(r"^(.+)combined_mask\.png$", filename)
    if match:
        prefix = match.group(1)          # e.g. "field_01_"
        groups[prefix].append(os.path.join(INPUT_DIR, filename))
    else:
        print(f"[SKIP] 패턴 불일치: {filename}")

if not groups:
    print("처리할 파일이 없습니다. 경로 및 파일명 패턴을 확인하세요.")
    exit(1)

print(f"총 {len(groups)}개 그룹 발견\n")

# 팽창용 커널 (원형, 5px 반경)
kernel_size = 2 * DILATE_PX + 1          # 11 × 11
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
)

for prefix, file_paths in sorted(groups.items()):
    print(f"[{prefix}]  파일 {len(file_paths)}개 처리 중...")
    for p in file_paths:
        print(f"  - {os.path.basename(p)}")

    combined = None
    for path in file_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ※ 읽기 실패: {path}")
            continue
        # 이진화 (0 or 255)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if combined is None:
            combined = np.zeros_like(binary)
        combined = cv2.bitwise_or(combined, binary)

    if combined is None:
        print(f"  ※ 유효한 마스크 없음, 건너뜀\n")
        continue

    dilated = cv2.dilate(combined, kernel, iterations=1)

    # 저장
    out_name = f"{prefix}mask.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, dilated)
    print(f"  → 저장 완료: {out_path}\n")

print("모든 처리 완료!")