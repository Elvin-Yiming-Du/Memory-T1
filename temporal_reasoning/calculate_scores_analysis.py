import json
import csv
from collections import defaultdict
from typing import List, Set, Dict
import re
from datetime import datetime, timedelta
from dateutil import parser

# 添加 PromptLengthAnalyzer 类
class PromptLengthAnalyzer:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.length_data = self._load_csv_data()
    
    def _load_csv_data(self) -> Dict[str, Dict]:
        data = {}
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data[row['question']] = {
                        'token_count': int(row['token_count']),
                        'length_category': row['length_category'],
                    }
            print(f"成功加载 {len(data)} 条问题数据")
        except FileNotFoundError:
            print(f"警告: CSV文件 {self.csv_file_path} 未找到")
            data = {}
        except Exception as e:
            print(f"加载CSV文件时出错: {e}")
            data = {}
        return data
    
    def get_length_category(self, question: str) -> str:
        info = self.length_data.get(question)
        return info['length_category'] if info else "unknown"

# 原有的函数保持不变
def split_items(s):
    s = s.strip()
    if s.startswith(('(', '（')) and s.endswith((')', '）')):
        s = s[1:-1]
    pattern = r'\)\s*\(|\)\s*\n\s*\(|）\s*（|）\s*\n\s*（'
    items = re.split(pattern, s)
    return items

def parse_time_expression_to_days(s: str) -> float:
    unit_map = {
        'year': 365.0, 'years': 365.0, 'month': 30.0, 'months': 30.0,
        'day': 1.0, 'days': 1.0, 'hour': 1.0 / 24, 'hours': 1.0 / 24,
        'minute': 1.0 / (24 * 60), 'minutes': 1.0 / (24 * 60),
        'second': 1.0 / (24 * 60 * 60), 'seconds': 1.0 / (24 * 60 * 60),
    }
    total_days = 0.0
    pattern = r'(\d+)\s+(year|years|month|months|day|days|hour|hours|minute|minutes|second|seconds)'
    matches = re.findall(pattern, s.lower())
    for value, unit in matches:
        value = int(value)
        if unit in unit_map:
            total_days += value * unit_map[unit]
    return total_days

def normalize_date(date_str):
    try:
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        try:
            dt = datetime.strptime(date_str, '%B %d, %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%b %d, %Y', '%d %B %Y'
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str
    except Exception:
        return date_str

def extract_option(answer):
    match = re.search(r'^[A-Z][\.\)\s]|^[A-Z]$', answer)
    if match:
        return match.group()[0]
    return answer

def extract_options(ans: str) -> Set[str]:
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
        "precision": precision, "recall": recall, "f1_score": f1,
        "exact_match": exact_match, "TP": tp, "FP": fp, "FN": fn
    }

def calculate_timeline_metrics(pred_str: str, gold_str: str) -> dict:
    pred = split_items(pred_str[1:-1])
    gold = split_items(gold_str[1:-1])
    tp = sum(1 for p, g in zip(pred, gold) if p == g)
    fp = len(pred) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred == gold else 0.0
    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "exact_match": exact_match, "TP": tp, "FP": fp, "FN": fn
    }

def calculate_localization_metrics(p_str: str, g_str: str) -> dict:
    if len(p_str) == 0 or len(g_str) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "exact_match": 0.0, "TP": 0, "FP": 0, "FN": 0}
    
    tp, fp, fn = 0, 0, 0
    p_valid, g_valid = False, False
    p_dt, g_dt = None, None

    try:
        p_dt = parser.parse(p_str)
        p_valid = True
    except:
        pass

    try:
        g_dt = parser.parse(g_str)
        g_valid = True
    except:
        pass

    if p_valid and g_valid:
        delta = abs(p_dt - g_dt)
        if delta <= timedelta(days=1):
            tp = 1
        else:
            fp = 1
            fn = 1
    else:
        if p_valid: fp = 1
        if g_valid: fn = 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if (tp == 1 and fp == 0 and fn == 0) else 0.0

    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "exact_match": exact_match, "TP": tp, "FP": fp, "FN": fn
    }

def calculate_computation_metrics(pred: str, gold: str) -> dict:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    pred_exprs = [' '.join(pred_tokens[i:i+2]) for i in range(0, len(pred_tokens), 2)]
    gold_exprs = [' '.join(gold_tokens[i:i+2]) for i in range(0, len(gold_tokens), 2)]

    tp = 0
    matched_gold_indices = set()
    for p_str in pred_exprs:
        p_days = parse_time_expression_to_days(p_str)
        found_match = False
        for g_idx, g_str in enumerate(gold_exprs):
            if g_idx in matched_gold_indices: continue
            g_days = parse_time_expression_to_days(g_str)
            if abs(p_days - g_days) <= 1.0:
                tp += 1
                matched_gold_indices.add(g_idx)
                found_match = True
                break

    fp = len(pred_exprs) - tp
    fn = len(gold_exprs) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if (tp == 1 and fp == 0 and fn == 0 and len(pred_exprs) == 1 and len(gold_exprs) == 1) else 0.0

    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "exact_match": exact_match, "TP": tp, "FP": fp, "FN": fn
    }

def compute_accuracy_by_task_and_length(filepath: str, length_csv_path: str, output_csv: str = None):
    # 加载长度分析器
    length_analyzer = PromptLengthAnalyzer(length_csv_path)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按任务和长度类别分组
    task_grouped = defaultdict(list)
    length_grouped = defaultdict(list)
    
    for i, item in enumerate(data):
        task = item.get("task", "Unknown")
        question = item.get("question", "")
        pred_ans = extract_option(item.get("parsed_output", {}).strip())
        gold_ans = item.get("gold_answer", "").strip()
        
        pred_ans = normalize_date(pred_ans)
        gold_ans = normalize_date(gold_ans)
        
        # 获取长度类别
        length_category = length_analyzer.get_length_category(question)
        
        task_grouped[task].append((pred_ans, gold_ans, item, length_category))
        length_grouped[length_category].append((pred_ans, gold_ans, item, task))

    print("=" * 80)
    print("PER TASK METRICS")
    print("=" * 80)

    output_order = ['Localization', 'Duration_Compare', 'Computation', 'Order_Compare', 
                   'Extract', 'Explicit_Reasoning', 'Order_Reasoning', 'Relative_Reasoning', 
                   'Counterfactual', 'Co_temporality', 'Timeline']
    
    task_results = {}
    length_results = defaultdict(lambda: defaultdict(list))
    
    all_f1, all_prec, all_recall, all_em = [], [], [], []
    all_TP, all_FP, all_FN = 0, 0, 0
    failure_cases = []
    
    # 按任务分析
    for task, items in task_grouped.items():
        task_prec, task_recall, task_f1, task_em = [], [], [], []
        
        for idx, (pred, gold, raw_item, length_category) in enumerate(items):
            pred_set = extract_options(pred)
            gold_set = extract_options(gold)
            
            if task == "Computation":
                metrics = calculate_computation_metrics(pred, gold)
            elif task == "Timeline":
                metrics = calculate_timeline_metrics(pred, gold)
            elif task == "Localization":
                metrics = calculate_localization_metrics(pred, gold)
            else:
                metrics = calculate_metrics(pred_set, gold_set)

            # 存储到任务结果
            task_prec.append(metrics["precision"])
            task_recall.append(metrics["recall"])
            task_f1.append(metrics["f1_score"])
            task_em.append(metrics["exact_match"])

            # 存储到长度类别结果
            length_results[length_category][task].append(metrics)

            # 全局统计
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
                    "length_category": length_category,
                    "question": raw_item.get("question", ""),
                    "gold": gold,
                    "predicted": pred,
                    "f1_score": metrics["f1_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"]
                })
        
        task_results[task] = {
            "samples": len(items),
            "exact_match": sum(task_em)/len(task_em) if task_em else 0,
            "f1_score": sum(task_f1)/len(task_f1) if task_f1 else 0,
            "precision": sum(task_prec)/len(task_prec) if task_prec else 0,
            "recall": sum(task_recall)/len(task_recall) if task_recall else 0
        }
        
        print(f"\nTask: {task}")
        print(f"  Samples: {len(items)}")
        print(f"  Exact Match: {task_results[task]['exact_match']:.4f}")
        print(f"  F1 Score: {task_results[task]['f1_score']:.4f}")
        print(f"  Precision: {task_results[task]['precision']:.4f}")
        print(f"  Recall: {task_results[task]['recall']:.4f}")

    # 按长度类别分析
    print("\n" + "=" * 80)
    print("PER LENGTH CATEGORY METRICS")
    print("=" * 80)
    
    length_category_results = {}
    for length_category, task_metrics in length_results.items():
        category_f1, category_prec, category_recall, category_em = [], [], [], []
        category_samples = 0
        
        for task, metrics_list in task_metrics.items():
            for metrics in metrics_list:
                category_f1.append(metrics["f1_score"])
                category_prec.append(metrics["precision"])
                category_recall.append(metrics["recall"])
                category_em.append(metrics["exact_match"])
                category_samples += 1
        
        length_category_results[length_category] = {
            "samples": category_samples,
            "exact_match": sum(category_em)/len(category_em) if category_em else 0,
            "f1_score": sum(category_f1)/len(category_f1) if category_f1 else 0,
            "precision": sum(category_prec)/len(category_prec) if category_prec else 0,
            "recall": sum(category_recall)/len(category_recall) if category_recall else 0
        }
        
        print(f"\nLength Category: {length_category}")
        print(f"  Samples: {category_samples}")
        print(f"  Exact Match: {length_category_results[length_category]['exact_match']:.4f}")
        print(f"  F1 Score: {length_category_results[length_category]['f1_score']:.4f}")
        print(f"  Precision: {length_category_results[length_category]['precision']:.4f}")
        print(f"  Recall: {length_category_results[length_category]['recall']:.4f}")

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"  Total Samples: {len(all_em)}")
    print(f"  Exact Match: {sum(all_em)/len(all_em):.4f}")
    print(f"  F1 Score: {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Precision: {sum(all_prec)/len(all_prec):.4f}")
    print(f"  Recall: {sum(all_recall)/len(all_recall):.4f}")
    print(f"  TP: {all_TP}, FP: {all_FP}, FN: {all_FN}")
    
    # 输出到CSV文件
    if output_csv:
        # 任务结果
        task_csv = output_csv
        with open(task_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Category', 'Type', 'F1 Score', 'Samples', 'Exact Match', 'Precision', 'Recall'])
            
            # 按任务写入
            for task in output_order:
                if task in task_results:
                    result = task_results[task]
                    writer.writerow(['Task', task, f"{result['f1_score']:.4f}", 
                                   result['samples'], f"{result['exact_match']:.4f}",
                                   f"{result['precision']:.4f}", f"{result['recall']:.4f}"])
            
            # 按长度类别写入
            for length_category in sorted(length_category_results.keys()):
                result = length_category_results[length_category]
                writer.writerow(['Length', length_category, f"{result['f1_score']:.4f}", 
                               result['samples'], f"{result['exact_match']:.4f}",
                               f"{result['precision']:.4f}", f"{result['recall']:.4f}"])
            
            # 总体结果
            writer.writerow(['Overall', 'All', f"{sum(all_f1)/len(all_f1):.4f}", 
                           len(all_em), f"{sum(all_em)/len(all_em):.4f}",
                           f"{sum(all_prec)/len(all_prec):.4f}", f"{sum(all_recall)/len(all_recall):.4f}"])
        
        print(f"\nExported results to: {task_csv}")
    
    # 导出失败案例（包含长度类别信息）
    if failure_cases:
        failure_csv = output_csv.replace('.csv', '_failures.csv') if output_csv else 'failures.csv'
        with open(failure_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=failure_cases[0].keys())
            writer.writeheader()
            writer.writerows(failure_cases)
        print(f"Exported {len(failure_cases)} failures to: {failure_csv}")

# 使用示例
compute_accuracy_by_task_and_length(
    filepath="/home/ma-user/work/ydu/temporal_reasoning/results/MemAgent-results_64k_0906.json",
    length_csv_path="/home/ma-user/work/ydu/data/time_range/prompt_length_analysis.csv",
    output_csv="/home/ma-user/work/ydu/temporal_reasoning/results/time_rangeMemAgent-results_64k_0906_analysis.csv"
)