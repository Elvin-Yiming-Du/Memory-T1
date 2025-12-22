# import json

# def compute_accuracy(filepath: str):
#     with open(filepath, 'r') as f:
#         data = json.load(f)
    
#     total = len(data)
#     correct = 0

#     for item in data:
#         predicted = item.get("parsed_output", {}).get("answer", "").strip()
#         gold = item.get("gold_answer", "").strip()
#         if predicted == gold:
#             correct += 1

#     accuracy = correct / total if total > 0 else 0
#     print(f"Total: {total}")
#     print(f"Correct: {correct}")
#     print(f"Accuracy: {accuracy:.4f}")

# # 使用路径运行
# compute_accuracy("/home/ubuntu/cloud_disk/temporal_reasoning/bm25_base_results.json")



import json
import csv
from collections import defaultdict
from typing import List, Set
import re
from datetime import datetime

def normalize_date(date_str):
    """将不同格式的日期字符串统一为YYYY-MM-DD格式"""
    try:
        # 尝试解析ISO格式 (2022-07-16T11:13:00)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        
        # 尝试解析常见格式 (July 14, 2022)
        try:
            dt = datetime.strptime(date_str, '%B %d, %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
            
        # 尝试解析其他常见格式
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y', 
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%b %d, %Y',  # Jul 14, 2022
            '%d %B %Y',   # 14 July 2022
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        # 如果所有格式都解析失败，返回原字符串
        return date_str
        
    except Exception:
        return date_str


def extract_option(answer):
    # 匹配单个大写字母选项（A-Z），可能后面跟着点、括号或空格
    match = re.search(r'^[A-Z][\.\)\s]|^[A-Z]$', answer)
    if match:
        return match.group()[0]  # 只返回字母部分
    return answer  # 如果没有匹配到选项格式，返回原答案


def extract_options(ans: str) -> Set[str]:
    """提取答案中的选项，支持多选或顺序格式，如 A B C、(1)(3)(2)"""
    ans = ans.strip()
    if "(" in ans and ")" in ans:
        return set(ans.replace(")(", ") (").split())
    else:
        return set(ans.split())

def calculate_metrics(pred: Set[str], gold: Set[str]) -> dict:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match = int(pred == gold)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "exact_match": exact_match,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def compute_accuracy_by_task(filepath: str, output_csv: str = None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # with open("/home/ubuntu/cloud_disk/temporal_reasoning/bm25_base_results.json", 'r', encoding='utf-8') as f2:
    #     ref_data = json.load(f2)

    grouped = defaultdict(list)
    for i, (item, ref_item) in enumerate(zip(data, data)):
        task = ref_item.get("task", "Unknown")
        print(item)
        pred_ans = extract_option(item.get("parsed_output", {}).get("answer","").strip())
        print(pred_ans)
        gold_ans = item.get("gold_answer", "").strip()
                # 统一格式化日期
        pred_ans = normalize_date(pred_ans)
        gold_ans = normalize_date(gold_ans)
        
        grouped[task].append((pred_ans, gold_ans, item))

    print("=" * 80)
    print("PER TASK METRICS")
    print("=" * 80)

    all_f1 = []
    all_prec = []
    all_recall = []
    all_em = []
    all_TP, all_FP, all_FN = 0, 0, 0

    failure_cases = []

    for task, items in grouped.items():
        task_prec, task_recall, task_f1, task_em = [], [], [], []
        for idx, (pred, gold, raw_item) in enumerate(items):
            pred_set = extract_options(pred)
            gold_set = extract_options(gold)
            metrics = calculate_metrics(pred_set, gold_set)

            task_prec.append(metrics["precision"])
            task_recall.append(metrics["recall"])
            task_f1.append(metrics["f1_score"])
            task_em.append(metrics["exact_match"])

            all_TP += metrics["TP"]
            all_FP += metrics["FP"]
            all_FN += metrics["FN"]
            all_f1.append(metrics["f1_score"])
            all_prec.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            all_em.append(metrics["exact_match"])

            if metrics["exact_match"] == 0:
                failure_cases.append({
                    "task": task,
                    "question": raw_item.get("question", ""),
                    "gold": gold,
                    "predicted": pred,
                    "f1_score": metrics["f1_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"]
                })

        print(f"\nTask: {task}")
        print(f"  Samples: {len(items)}")
        print(f"  Exact Match: {sum(task_em)/len(task_em):.4f}")
        print(f"  F1 Score: {sum(task_f1)/len(task_f1):.4f}")
        print(f"  Precision: {sum(task_prec)/len(task_prec):.4f}")
        print(f"  Recall: {sum(task_recall)/len(task_recall):.4f}")

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"  Total Samples: {len(all_em)}")
    print(f"  Exact Match: {sum(all_em)/len(all_em):.4f}")
    print(f"  F1 Score: {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Precision: {sum(all_prec)/len(all_prec):.4f}")
    print(f"  Recall: {sum(all_recall)/len(all_recall):.4f}")
    print(f"  TP: {all_TP}, FP: {all_FP}, FN: {all_FN}")

    # 可选导出失败案例
    if output_csv and failure_cases:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=failure_cases[0].keys())
            writer.writeheader()
            writer.writerows(failure_cases)
        print(f"\nExported {len(failure_cases)} failures to: {output_csv}")

# 示例调用
compute_accuracy_by_task(
    filepath="/home/ma-user/work/ydu/temporal_reasoning/results/time_range_qwen2.5-14B-Instruct-base_0829_without_utt_time.json",
    output_csv="/home/ma-user/work/ydu/temporal_reasoning/results/time_range_qwen2.5-14B-Instruct-base_0829_without_utt_time.csv"
)