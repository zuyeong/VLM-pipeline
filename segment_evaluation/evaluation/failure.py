import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from collections import defaultdict

# === 경로 설정 === 재실행시 수정 필요
base_dir = '/home/cjy/workspace/segment/evaluation_dataset'
csv_path = os.path.join(base_dir, 'detailed_prompt_metrics.csv')
output_dir = os.path.join(base_dir, 'failure_visualizations')
os.makedirs(output_dir, exist_ok=True)

models = ['glamm', 'instructseg', 'pixellm', 'read', 'wise']
display_names = ['GT', 'GLaMM', 'InstructSeg', 'PixelLM', 'READ', 'WISE']

# === CSV 로드 ===
rows = []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# === IoU=0 이미지별 묶기 ===
image_prompts = defaultdict(list)
for r in rows:
    if float(r['IoU']) == 0.0:
        key = r['Image_Name']
        prompt = r['Prompt_Text']
        if prompt not in image_prompts[key]:
            image_prompts[key].append(prompt)

image_prompts = dict(sorted(image_prompts.items()))
print(f'이미지 수: {len(image_prompts)}, 총 프롬프트 조합: {sum(len(v) for v in image_prompts.values())}개')

# === 예측 마스크 경로 보정 ===
def resolve_pred_path(base_dir, model, raw_pred):
    candidates = []
    if f'results_{model}' in raw_pred:
        idx = raw_pred.find(f'results_{model}')
        candidates.append(os.path.join(base_dir, 'model_results', raw_pred[idx:]))
    candidates.append(os.path.join(base_dir, 'model_results', raw_pred))
    candidates.append(os.path.join(base_dir, 'model_results', f'results_{model}', os.path.basename(raw_pred)))
    candidates.append(os.path.join(base_dir, 'model_results', f'results_{model}', 'person', os.path.basename(raw_pred)))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# === 이미지별 시각화 ===
for img_idx, (target_img, prompts) in enumerate(image_prompts.items()):
    n_prompts = len(prompts)
    n_cols = 1 + len(models)  # 6열

    # 행 비율 설정: [프롬프트 행, 이미지 행] 반복
    hr = []
    for _ in range(n_prompts):
        hr.extend([0.15, 1.0])

    # hspace를 늘려 프롬프트와 모델명 사이 간격 확보
    fig, axes = plt.subplots(
        n_prompts * 2, n_cols,
        figsize=(0.9 * n_cols, 2.2 * n_prompts + 0.5),
        gridspec_kw={'height_ratios': hr, 'wspace': 0.1, 'hspace': 0.4}
    )

    if len(axes.shape) == 1:
        axes = axes.reshape(-1, n_cols)

    fig.suptitle(f'{target_img}', fontsize=12, y=1.02) # 볼드체 제거

    for row_idx, prompt in enumerate(prompts):
        text_row = row_idx * 2
        img_row = row_idx * 2 + 1

        # 프롬프트 행 축 숨기기
        for j in range(n_cols):
            axes[text_row][j].axis('off')

        matched = [r for r in rows if r['Image_Name'] == target_img and r['Prompt_Text'] == prompt]
        if not matched:
            continue

        gt_path = os.path.join(base_dir, matched[0]['GT_Path'])
        prompt_type = matched[0].get('Prompt_Type', '')
        fail_models = [r['Model'] for r in matched if float(r['IoU']) == 0.0]

        # 프롬프트 텍스트 삽입 (볼드체 제거)
        axes[text_row][0].text(
            0.0, 0.5, f'Prompt: "{prompt}"  [{prompt_type}]  (fail: {len(fail_models)}/5)',
            ha='left', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7F7F7', edgecolor='#CCCCCC', linewidth=0.8)
        )

        # GT 불러오기
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            for j in range(n_cols): axes[img_row][j].axis('off')
            continue
        h, w = gt_mask.shape

        # GT 표시
        gt_display = np.zeros((h, w, 3), dtype=np.uint8)
        gt_display[gt_mask > 0] = [255, 255, 255]

        ax_gt = axes[img_row][0]
        ax_gt.set_title(display_names[0], fontsize=6) # 이미지 바로 위에 모델명 (볼드체 제거)
        ax_gt.imshow(gt_display)
        
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values(): spine.set_visible(False)

        # 모델별 오버레이
        for i, model in enumerate(models):
            ax = axes[img_row][1 + i]
            ax.set_title(display_names[1 + i], fontsize=6) # 이미지 바로 위에 모델명 (볼드체 제거)

            model_row = [r for r in matched if r['Model'] == model]

            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[gt_mask > 0] = [0, 200, 0]

            iou_val = None
            if model_row:
                raw_pred = model_row[0]['Pred_Path']
                iou_val = float(model_row[0]['IoU'])
                pred_path = resolve_pred_path(base_dir, model, raw_pred)
                pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) if pred_path else None

                if pred_mask is not None:
                    pred_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_bin = pred_resized > 0
                    overlap = (gt_mask > 0) & pred_bin
                    pred_only = (~(gt_mask > 0)) & pred_bin
                    canvas[overlap] = [255, 220, 0]
                    canvas[pred_only] = [220, 0, 0]

            ax.imshow(canvas)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

            title_color = 'red' if iou_val == 0.0 else 'black'
            iou_str = f'{iou_val:.3f}' if iou_val is not None else 'N/A'
            # labelpad를 늘려 수치가 이미지와 너무 붙지 않게 조정
            ax.set_xlabel(iou_str, fontsize=6, color=title_color, labelpad=5)

    # 하단 여백을 충분히 확보하여 수치와 범례가 겹치지 않게 방지
    plt.subplots_adjust(bottom=0.15)

    # 색깔안내 (범례) 설정
    legend = [
        mpatches.Patch(color='#00C800', label='GT only (missed)'),
        mpatches.Patch(color='#FFDC00', label='Matched (TP)'),
        mpatches.Patch(color='#DC0000', label='Pred only (FP)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.02), frameon=False)

    save_path = os.path.join(output_dir, f'{img_idx+1:02d}_{target_img}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[{img_idx+1}/{len(image_prompts)}] Saved: {save_path} ({n_prompts} prompts)')

print(f'\n완료! 총 {len(image_prompts)}장 → {output_dir}')