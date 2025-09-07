import json
import csv
from collections import defaultdict
from typing import List, Set
import re
from datetime import datetime
from typing import List
from datetime import datetime, timedelta
from dateutil import parser

import re
def split_items(s):
    # 去除字符串首尾的空白字符
    s = s.strip()
    # 去除首尾的括号（英文或中文）
    if s.startswith(('(', '（')) and s.endswith((')', '）')):
        s = s[1:-1]
    
    # 使用正则表达式分割：匹配 )(、)\n(、）（、）\n（ 等变体
    # 模式解释：匹配右括号后跟可选空白（包括换行）和左括号
    pattern = r'\)\s*\(|\)\s*\n\s*\(|）\s*（|）\s*\n\s*（'
    items = re.split(pattern, s)
    return items

def parse_timeline_string(s: str) -> list[str]:
    matches = re.findall(r'$(\d+)$', s)
    return matches

def parse_time_expression_to_days(s: str) -> float:
    # 定义时间单位映射（近似值，简化计算，不考虑闰年/闰月）
    unit_map = {
        'year': 365.0,
        'years': 365.0,
        'month': 30.0,
        'months': 30.0,
        'day': 1.0,
        'days': 1.0,
        'hour': 1.0 / 24,
        'hours': 1.0 / 24,
        'minute': 1.0 / (24 * 60),
        'minutes': 1.0 / (24 * 60),
        'second': 1.0 / (24 * 60 * 60),
        'seconds': 1.0 / (24 * 60 * 60),
    }

    total_days = 0.0
    # 正则匹配数字 + 单位，例如 "22 days", "1 year", "12 hours"
    pattern = r'(\d+)\s+(year|years|month|months|day|days|hour|hours|minute|minutes|second|seconds)'
    matches = re.findall(pattern, s.lower())  # 统一小写匹配

    for value, unit in matches:
        value = int(value)
        if unit in unit_map:
            total_days += value * unit_map[unit]
        else:
            # 未知单位，忽略或报错（根据需求）
            pass

    return total_days

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


def calculate_timeline_metrics(pred_str: str, gold_str: str) -> dict:
    pred = split_items(pred_str[1:-1])
    gold = split_items(gold_str[1:-1])

    # --- Step 2: 顺序敏感评测指标计算 ---
    tp = sum(1 for p, g in zip(pred, gold) if p == g)
    fp = len(pred) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ✅ 推荐：直接比较两个列表是否完全一致（顺序 + 元素）
    exact_match = 1.0 if pred == gold else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "exact_match": exact_match,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }
    
    
def calculate_localization_metrics(p_str: str, g_str: str) -> dict:
    # pred_list = list(pred)
    # gold_list = list(gold)
    # print(pred_list)
    # print(gold_list)
    # 你说明过 pred 和 gold 都只含有一个元素，所以我们直接取第一个（也是唯一一个）
    if len(p_str) == 0 or len(g_str) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "exact_match": 0.0,
            "TP": 0,
            "FP": 0,
            "FN": 0
        }


    tp = 0
    fp = 0
    fn = 0

    try:
        p_dt = parser.parse(p_str)
        pred_valid = True
    except:
        pred_valid = False  # pred 无法解析，视为无效预测
    
#     print(p_dt)
    
    try:
        g_dt = parser.parse(g_str)
        gold_valid = True
    except:
        gold_valid = False  # gold 无法解析，视为无效真实值
        
    # print(g_dt)
    # 只有当 pred 和 gold 都能解析时，才进行时间差判断
    if pred_valid and gold_valid:
        delta = abs(p_dt - g_dt)
        if delta <= timedelta(days=2):  # 时间差在 1 天以内
            tp = 1
        else:
            fp = 1  # 预测了但时间差距过大
            fn = 1  # 真实存在但预测不匹配
    else:
        # 任意一个无法解析，就无法匹配
        fp = 1 if pred_valid else 0  # 如果 pred 合法但 gold 不合法，pred 也算 FP
        fn = 1 if gold_valid else 0  # 如果 gold 合法但 pred 不合法，gold 也算 FN

        # 下面是更细致的处理，可以根据需求调整
        if pred_valid and not gold_valid:
            fp = 1
            fn = 0
        elif not pred_valid and gold_valid:
            fp = 0
            fn = 1
        elif not pred_valid and not gold_valid:
            fp = 0
            fn = 0
            tp = 0

    # 修正更合理的 FP / FN 定义（推荐如下方式）
    if pred_valid and gold_valid:
        delta = abs(p_dt - g_dt)
        if delta <= timedelta(days=1):
            tp = 1
            fp = 0
            fn = 0
        else:
            tp = 0
            fp = 1  # 预测了但时间差超过 1 天
            fn = 1  # 真实存在但未正确预测
    else:
        # 有一个或两个都无法解析，无法匹配
        tp = 0
        fp = 1 if pred_valid else 0
        fn = 1 if gold_valid else 0

    # 更清晰的逻辑（推荐最终使用这个）
    tp = 0
    fp = 0
    fn = 0

    p_valid = False
    g_valid = False
    p_dt = None
    g_dt = None

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
        # 无法匹配的情况
        if p_valid:
            fp = 1  # 预测了但 gold 无效或时间差超限（这里简化：只要 gold 无效，pred也算FP）
        if g_valid:
            fn = 1  # 真实存在但 pred 无效或时间差超限

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_match = 1.0 if (tp == 1 and fp == 0 and fn == 0) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "exact_match": exact_match,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }
    
def calculate_computation_metrics(pred: str, gold: str) -> dict:
    # --- Step 1: 按空格分割成 tokens，然后每两个 token 组成一个时间表达式 ---
    pred_tokens = pred.split()  # 如 ['1', 'year', '3', 'months', ...]
    gold_tokens = gold.split()  # 如 ['3', 'months', '8', 'days', ...]

    # 每两个 token 组成一个时间表达式，如 ['1', 'year'] → '1 year'
    pred_exprs = [' '.join(pred_tokens[i:i+2]) for i in range(0, len(pred_tokens), 2)]
    gold_exprs = [' '.join(gold_tokens[i:i+2]) for i in range(0, len(gold_tokens), 2)]

    # --- Step 2: 匹配逻辑：对每个 pred，找一个未匹配的 gold，时间差 <= 1 天 ---
    tp = 0
    matched_gold_indices = set()

    for p_str in pred_exprs:
        p_days = parse_time_expression_to_days(p_str)
        found_match = False
        for g_idx, g_str in enumerate(gold_exprs):
            if g_idx in matched_gold_indices:
                continue
            g_days = parse_time_expression_to_days(g_str)
            if abs(p_days - g_days) <= 1.0:  # 时间差在 1 天以内
                # print(f"[MATCH] pred '{p_str}' ({p_days} days) ≈ gold '{g_str}' ({g_days} days)")
                tp += 1
                matched_gold_indices.add(g_idx)
                found_match = True
                break
        # 未找到匹配则为 FP

    fp = len(pred_exprs) - tp
    fn = len(gold_exprs) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_match = (
        1.0
        if (tp == 1 and fp == 0 and fn == 0 and len(pred_exprs) == 1 and len(gold_exprs) == 1)
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "exact_match": exact_match,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
    

def compute_accuracy_by_task(filepath: str, output_csv: str = None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # with open("/home/ubuntu/cloud_disk/temporal_reasoning/bm25_base_results.json", 'r', encoding='utf-8') as f2:
    #     ref_data = json.load(f2)

    grouped = defaultdict(list)
    for i, (item, ref_item) in enumerate(zip(data, data)):
        task = ref_item.get("task", "Unknown")
        pred_ans = extract_option(item.get("parsed_output", {}).strip())
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
            # print("-"*50)
            # print(pred)
            # print(gold)
            # print("-"*50)
            if task == "Computation":
              
                metrics = calculate_computation_metrics(pred, gold)
            elif task == "Timeline":
               
                metrics = calculate_timeline_metrics(pred, gold)
            elif task == "Localization":
                metrics = calculate_localization_metrics(pred, gold)
            else:
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
    filepath="/home/ma-user/work/ydu/temporal_reasoning/results/Qwen2.5_7B_Instruct-results_32k_0905.json",
    output_csv="/home/ma-user/work/ydu/temporal_reasoning/results/long_context_Qwen2.5_7B_Instruct-results_32k_0905-results.csv"
)