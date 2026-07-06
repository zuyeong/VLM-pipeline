import os
import csv
import glob
import cv2
import numpy as np
from collections import defaultdict

def get_iou(im, gt):
    intersection = np.logical_and(gt > 0, im > 0).sum()
    union = np.logical_or(gt > 0, im > 0).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_precision(im, gt):
    TPFP = np.sum(im != 0)
    TP = np.sum((gt != 0) & (im != 0))
    if TPFP == 0:
        return 0.0
    return TP / TPFP

def get_recall(im, gt):
    TPFN = np.sum(gt != 0)
    TP = np.sum((gt != 0) & (im != 0))
    if TPFN == 0:
        return 0.0
    return TP / TPFN

def get_f1(im, gt):
    p = get_precision(im, gt)
    r = get_recall(im, gt)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def get_boundary_iou(im, gt, dilation_ratio=0.02):
    img_diag = np.sqrt(gt.shape[0]**2 + gt.shape[1]**2)
    d = int(round(dilation_ratio * img_diag))
    if d < 1: d = 1

    def get_boundary(mask):
        mask_binary = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        boundary = mask_binary - eroded
        dilated_boundary = cv2.dilate(boundary, kernel, iterations=d)
        return dilated_boundary

    gt_boundary = get_boundary(gt)
    pred_boundary = get_boundary(im)

    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

base_dir = '/home/cjy/workspace/segment/evaluation_dataset'
models = ['glamm', 'instructseg', 'pixellm', 'read', 'wise']

print("Calculating overall and prompt-type detailed metrics for all models...")

all_detailed_rows = []
summary_results = {}
prompt_analysis = defaultdict(lambda: defaultdict(lambda: {'iou': [], 'f1': [], 'biou': [], 'zeros': 0}))

for model in models:
    csv_pattern = os.path.join(base_dir, 'model_results', f'results_{model}', 'pred_results_log*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print(f"CSV not found for model {model}")
        continue
    csv_path = csv_files[0]
    
    model_metrics = {'iou': [], 'f1': [], 'biou': [], 'zeros': 0}
        
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_path = os.path.join(base_dir, row['gt_mask_path'])
            raw_pred = row['pred_mask_path']
            ptype = row.get('prompt_type', 'unknown')
            
            if f'results_{model}' in raw_pred:
                idx = raw_pred.find(f'results_{model}')
                fixed_rel_path = raw_pred[idx:]
                pred_path = os.path.join(base_dir, 'model_results', fixed_rel_path)
            else:
                pred_path = os.path.join(base_dir, 'model_results', raw_pred)
                
            if not os.path.exists(pred_path):
                alt_pred_path = os.path.join(base_dir, 'model_results', f'results_{model}', os.path.basename(raw_pred))
                if os.path.exists(alt_pred_path):
                    pred_path = alt_pred_path
                else:
                    alt_pred_path2 = os.path.join(base_dir, 'model_results', f'results_{model}', 'person', os.path.basename(raw_pred))
                    if os.path.exists(alt_pred_path2):
                        pred_path = alt_pred_path2
                    else:
                        continue
                        
            im = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if im is None or gt is None:
                continue
                
            iou = get_iou(im, gt)
            f1 = get_f1(im, gt)
            biou = get_boundary_iou(im, gt)
            
            # Overall model tracking
            model_metrics['iou'].append(iou)
            model_metrics['f1'].append(f1)
            model_metrics['biou'].append(biou)
            if iou == 0.0:
                model_metrics['zeros'] += 1
            
            # Prompt type tracking
            prompt_analysis[model][ptype]['iou'].append(iou)
            prompt_analysis[model][ptype]['f1'].append(f1)
            prompt_analysis[model][ptype]['biou'].append(biou)
            if iou == 0.0:
                prompt_analysis[model][ptype]['zeros'] += 1
            
            # Save detailed row
            all_detailed_rows.append({
                'Model': model,
                'Image_Name': row.get('image_name', ''),
                'Prompt_Type': ptype,
                'Prompt_Text': row.get('prompt_text', ''),
                'IoU': round(iou, 4),
                'F1_Score': round(f1, 4),
                'Boundary_IoU': round(biou, 4),
                'GT_Path': row['gt_mask_path'],
                'Pred_Path': raw_pred
            })
            
    m_iou = np.mean(model_metrics['iou']) if model_metrics['iou'] else 0
    m_f1 = np.mean(model_metrics['f1']) if model_metrics['f1'] else 0
    m_biou = np.mean(model_metrics['biou']) if model_metrics['biou'] else 0
    
    summary_results[model] = {
        'count': len(model_metrics['iou']),
        'zeros': model_metrics['zeros'],
        'mIoU': m_iou,
        'F1-score': m_f1,
        'Boundary IoU': m_biou
    }

# --- 1. Print Overall Summary ---
print("\n" + "="*85)
print("[1] OVERALL MODEL PERFORMANCE")
print("="*85)
print(f"{'Model':<15} | {'mIoU':<10} | {'F1-Score':<10} | {'Boundary IoU':<15} | {'Failure Rate':<15}")
print("-" * 85)
for m in models:
    if m in summary_results:
        res = summary_results[m]
        fail_rate = (res['zeros'] / res['count']) if res['count'] > 0 else 0
        print(f"{m:<15} | {res['mIoU']:.4f}     | {res['F1-score']:.4f}     | {res['Boundary IoU']:.4f}          | {fail_rate:.4f} ({res['zeros']}/{res['count']})")
print("="*85)

# --- 2. Print Prompt Type Summary ---
print("\n" + "="*85)
print("[2] PERFORMANCE BY PROMPT TYPE")
print("="*85)
print(f"{'Model':<15} | {'Prompt Type':<12} | {'mIoU':<8} | {'F1-Score':<8} | {'B-IoU':<8} | {'Failure Rate':<12}")
print("-" * 85)

prompt_output_rows = []
for model in sorted(prompt_analysis.keys()):
    for ptype in sorted(prompt_analysis[model].keys()):
        stats = prompt_analysis[model][ptype]
        count = len(stats['iou'])
        
        m_iou = sum(stats['iou']) / count if count > 0 else 0
        m_f1 = sum(stats['f1']) / count if count > 0 else 0
        m_biou = sum(stats['biou']) / count if count > 0 else 0
        zeros = stats['zeros']
        
        fail_rate = (zeros / count) if count > 0 else 0
        fail_str = f"{fail_rate:.4f}"
        
        print(f"{model:<15} | {ptype:<12} | {m_iou:.4f}   | {m_f1:.4f}   | {m_biou:.4f}   | {fail_str:>12}")
        
        prompt_output_rows.append({
            'Model': model,
            'Prompt_Type': ptype,
            'mIoU': round(m_iou, 4),
            'F1_Score': round(m_f1, 4),
            'Boundary_IoU': round(m_biou, 4),
            'Failure_Rate': round(fail_rate, 4)
        })
    print("-" * 85)
print("="*85)

# --- 3. Save CSVs ---
out_csv_path_detailed = os.path.join(base_dir, 'detailed_prompt_metrics.csv')
if all_detailed_rows:
    fieldnames = ['Model', 'Image_Name', 'Prompt_Type', 'Prompt_Text', 'IoU', 'F1_Score', 'Boundary_IoU', 'GT_Path', 'Pred_Path']
    with open(out_csv_path_detailed, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detailed_rows)

out_csv_path_ptype = os.path.join(base_dir, 'prompt_type_analysis.csv')
if prompt_output_rows:
    fieldnames = ['Model', 'Prompt_Type', 'mIoU', 'F1_Score', 'Boundary_IoU', 'Failure_Rate']
    with open(out_csv_path_ptype, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prompt_output_rows)

print(f"\n✅ 모든 데이터 처리가 완료되었습니다.")
print(f"1) 모든 프롬프트별 상세 평가 로그: {out_csv_path_detailed}")
print(f"2) 프롬프트 유형별 요약 통계: {out_csv_path_ptype}")
