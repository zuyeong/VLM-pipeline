import os
import csv
from collections import defaultdict

csv_path = '/home/cjy/workspace/segment/evaluation_dataset/detailed_prompt_metrics.csv'
out_csv_path = '/home/cjy/workspace/segment/evaluation_dataset/prompt_type_analysis.csv'

# 데이터 구조: analysis[model][prompt_type] = {'iou': [], 'f1': [], 'biou': [], 'zeros': 0}
analysis = defaultdict(lambda: defaultdict(lambda: {'iou': [], 'f1': [], 'biou': [], 'zeros': 0}))

# 1. 상세 결과 CSV 파일 읽기
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row['Model']
        ptype = row['Prompt_Type']
        
        # 값이 비어있거나 잘못된 경우 예외 처리
        try:
            iou = float(row['IoU'])
            f1 = float(row['F1_Score'])
            biou = float(row['Boundary_IoU'])
        except ValueError:
            continue
        
        analysis[model][ptype]['iou'].append(iou)
        analysis[model][ptype]['f1'].append(f1)
        analysis[model][ptype]['biou'].append(biou)
        
        # IoU가 정확히 0.0이면 인물을 아예 인식하지 못한 경우 (실패)
        if iou == 0.0:
            analysis[model][ptype]['zeros'] += 1

output_rows = []

print("="*90)
print(f"{'Model':<15} | {'Prompt Type':<12} | {'mIoU':<8} | {'F1-Score':<8} | {'B-IoU':<8} | {'0점(실패)/총 개수'}")
print("-" * 90)

# 2. 통계 계산 및 출력
for model in sorted(analysis.keys()):
    for ptype in sorted(analysis[model].keys()):
        stats = analysis[model][ptype]
        count = len(stats['iou'])
        
        m_iou = sum(stats['iou']) / count if count > 0 else 0
        m_f1 = sum(stats['f1']) / count if count > 0 else 0
        m_biou = sum(stats['biou']) / count if count > 0 else 0
        zeros = stats['zeros']
        
        # 터미널 출력용 포맷팅
        fail_str = f"{zeros} / {count}"
        print(f"{model:<15} | {ptype:<12} | {m_iou:.4f}   | {m_f1:.4f}   | {m_biou:.4f}   | {fail_str:>15}")
        
        output_rows.append({
            'Model': model,
            'Prompt_Type': ptype,
            'mIoU': round(m_iou, 4),
            'F1_Score': round(m_f1, 4),
            'Boundary_IoU': round(m_biou, 4),
            'Failure_Count(IoU=0)': zeros,
            'Total_Count': count
        })
    print("-" * 90)

print("="*90)

# 3. 분석 결과를 새로운 CSV 파일로 저장
with open(out_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    fieldnames = ['Model', 'Prompt_Type', 'mIoU', 'F1_Score', 'Boundary_IoU', 'Failure_Count(IoU=0)', 'Total_Count']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\\n✅ 프롬프트 유형(Prompt Type)별 분석 결과가 엑셀 파일로 저장되었습니다:")
print(f"-> {out_csv_path}")
