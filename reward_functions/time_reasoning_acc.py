from datetime import datetime, date
import math
import json
import ast  # 使用 ast.literal_eval 来安全地解析Python字面量
import re
from collections import defaultdict
import math
import re
import re
from collections import defaultdict
import json
import ast
import re
from collections import defaultdict
from dateutil import parser
from datetime import timedelta

# ------------------------- 从评估脚本中集成的辅助函数 -------------------------
def parse_time_expression_to_days(s: str) -> float:
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
    pattern = r'(\d+)\s+(year|years|month|months|day|days|hour|hours|minute|minutes|second|seconds)'
    matches = re.findall(pattern, s.lower())
    for value, unit in matches:
        value = int(value)
        if unit in unit_map:
            total_days += value * unit_map[unit]
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

def extract_options(ans: str):
    ans = ans.strip()
    if "(" in ans and ")" in ans:
        return set(ans.replace(")(", ") (").split())
    else:
        return set(ans.split())

def calculate_metrics(pred: set, gold: set):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match = int(pred == gold)
    return {"f1_score": f1, "exact_match": exact_match, "precision": precision, "recall": recall, "TP": tp, "FP": fp, "FN": fn}

def calculate_timeline_metrics(pred_str: str, gold_str: str):
    pred = pred_str[1:-1].split(")(") if pred_str.startswith('(') and pred_str.endswith(')') else []
    gold = gold_str[1:-1].split(")(") if gold_str.startswith('(') and gold_str.endswith(')') else []
    tp = sum(1 for p, g in zip(pred, gold) if p == g)
    fp = len(pred) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred == gold else 0.0
    return tp/len(pred)

def calculate_localization_metrics(norm_pred, norm_gold):
    # 尝试解析规范化后的日期
    try:
        pred_dt = datetime.strptime(norm_pred, '%Y-%m-%d')
        gold_dt = datetime.strptime(norm_gold, '%Y-%m-%d')
        
        # 计算日期差异
        delta = abs((pred_dt - gold_dt).days)
        
        # 基于差异计算奖励
        if delta == 0:
            # 完全匹配，最高奖励
            return 1.0
        elif delta <= 7:
            # 一周内的差异，奖励逐渐减少
            return max(0.5, 1.0 - (delta * 0.1))
        elif delta <= 30:
            # 一个月内的差异，奖励较低
            return max(0.1, 0.5 - ((delta - 7) * 0.02))
        else:
            # 超过一个月的差异，最低奖励
            return 0.0
            
    except (ValueError, TypeError):
        # 如果日期无法解析，检查字符串是否相同
        if norm_pred == norm_gold:
            return 1.0
        else:
            return 0.0

def calculate_computation_metrics(pred: str, gold: str):
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
            if g_idx in matched_gold_indices:
                continue
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
    exact_match = 1.0 if (tp == len(gold_exprs) and fp == 0 and fn == 0) else 0.0
    return {"f1_score": f1, "exact_match": exact_match, "precision": precision, "recall": recall, "TP": tp, "FP": fp, "FN": fn}

def simple_text_match(short_text, long_text):
    """
    简单的短文片段与长文本匹配算法
    
    参数:
    short_text (str): 短文片段（如事件文本）
    long_text (str): 长文本（如问题文本）
    
    返回:
    float: 匹配分数，范围0-1
    """
    if not short_text or not long_text:
        return 0.0
    
    # 将文本转换为小写并分词
    short_words = re.findall(r'\w+', short_text.lower())
    long_words = re.findall(r'\w+', long_text.lower())
    
    # 移除停用词
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'}
    short_words = [word for word in short_words if word not in stop_words and len(word) > 2]
    long_words = [word for word in long_words if word not in stop_words and len(word) > 2]
    
    if not short_words:
        return 0.0
    
    # 计算词重叠率
    short_set = set(short_words)
    long_set = set(long_words)
    
    # 基本重叠率
    overlap = len(short_set.intersection(long_set)) / len(short_set)
    
    # 考虑词序和位置权重
    position_score = 0.0
    if overlap > 0:
        # 找到匹配词在长文本中的位置
        matched_positions = []
        for i, word in enumerate(long_words):
            if word in short_set:
                matched_positions.append(i)
        
        # 计算位置连续性（连续匹配的奖励）
        continuity_bonus = 0.0
        if len(matched_positions) > 1:
            # 计算连续匹配的长度
            max_continuity = 1
            current_continuity = 1
            for i in range(1, len(matched_positions)):
                if matched_positions[i] == matched_positions[i-1] + 1:
                    current_continuity += 1
                    max_continuity = max(max_continuity, current_continuity)
                else:
                    current_continuity = 1
            
            # 连续匹配奖励
            continuity_bonus = min(0.3, max_continuity / len(short_words) * 0.3)
        
        # 位置权重（匹配词越靠前，权重越高）
        position_weight = 0.0
        if matched_positions:
            # 计算平均位置（归一化到0-1）
            avg_position = sum(matched_positions) / len(matched_positions) / len(long_words)
            position_weight = (1 - avg_position) * 0.2  # 靠前的词权重更高
        
        position_score = continuity_bonus + position_weight
    
    # 综合得分
    match_score = overlap + position_score
    
    return min(1.0, match_score)  # 确保不超过1


def extract_question_content(template_text):
    """
    从模板文本中抽取"Question: "后面的内容
    
    参数:
    template_text (str): 包含问题模板的文本
    
    返回:
    str: 提取到的问题内容，如果未找到则返回空字符串
    """
    # 使用正则表达式匹配"Question: "后面的内容
    pattern = r"Question:\s*(.*?)(?=\n</question>|\Z)"
    match = re.search(pattern, template_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # 如果第一种模式没匹配到，尝试更宽松的匹配
        pattern_alt = r"Question:\s*(.*)"
        match_alt = re.search(pattern_alt, template_text, re.DOTALL)
        if match_alt:
            return match_alt.group(1).strip()
        else:
            return ""


def parse_stored_time_range(time_range_str):
    """解析使用 str() 存储的时间范围字符串"""
    if not time_range_str or not isinstance(time_range_str, str):
        return {}
    
    try:
        # 使用 ast.literal_eval 来解析Python字面量
        return ast.literal_eval(time_range_str)
    except (ValueError, SyntaxError):
        try:
            # 如果上面失败，尝试替换单引号为双引号然后用json解析
            fixed_str = time_range_str.replace("'", '"')
            import json
            return json.loads(fixed_str)
        except:
            return {}
        
def parse_timestamp_with_default(timestamp, condition):
    """
    解析时间戳字符串，如果是"unknown"则根据条件返回默认值
    
    Args:
        timestamp: 时间戳字符串，可能是ISO格式或"unknown"
        condition: 条件值，决定使用哪个默认值
    
    Returns:
        datetime.date对象或默认日期字符串
    """
    if isinstance(timestamp, date):
        return timestamp
    
    if timestamp is None or str(timestamp).lower() == "unknown":
        # 根据条件返回不同的默认日期
        if condition == 1:
            return date(2020, 1, 1)  # 或者返回 datetime(1900, 1, 1).date()
        else:
            return date(2025, 8, 1)  # 或者返回 datetime(2300, 12, 31).date()
    else:
        # 解析时间戳并返回日期部分
        try:
            # 处理带时区和不带时区的情况
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # 如果只有日期部分
                dt = datetime.strptime(timestamp, "%Y-%m-%d")
            return dt.date()  # 返回date对象！
        except ValueError:
            # 如果解析失败，也返回默认值
            if condition == 1:
                return date(2020, 1, 1)
            else:
                return date(2025, 8, 1)

def check_time_range_relation(st, et, est, eet):

    # 检查时间区间是否有效
    if st > et or est > eet:
        return -1
    
    # 判断区间关系
    # 1. 完全包含：第一个区间完全包含第二个区间
    if st <= est and et >= eet:
        return 1
    
    # 2. 完全不交叉：第一个区间在第二个区间之前或之后
    if et < est or st > eet:
        return -1
    
    # 3. 有交叉：其他情况
    return 0.5

def calculate_session_time_reward(session_time_parsed, start_time, end_time):
    """
    基于相对距离的自适应时间奖励函数
    无需手动调整超参数，自动适应不同长度的时间范围
    """
    # 计算时间范围的长度（以天为单位）
    range_length_days = (end_time - start_time).days
    # 确保范围长度至少为1天，避免除零
    range_length_days = max(range_length_days, 1)
    
    # 计算session时间到时间范围最近边界的距离
    if session_time_parsed < start_time:
        distance_to_range = (start_time - session_time_parsed).days
    elif session_time_parsed > end_time:
        distance_to_range = (session_time_parsed - end_time).days
    else:
        # 在时间范围内，给予最高奖励
        return 1.0
    
    # 关键改进：使用相对距离而不是绝对距离
    # 距离相对于时间范围长度的比例
    relative_distance = distance_to_range / range_length_days
    
    # 使用sigmoid函数创建平滑的奖励衰减
    # 当relative_distance=0时（刚好在边界），奖励接近1
    # 当relative_distance增大时，奖励平滑衰减到-0.2左右
    def sigmoid_reward(x):
        # 将相对距离映射到奖励值
        # 参数经过精心选择，无需调整
        return 1.5 / (1 + math.exp(0.5 * x)) - 0.5
    try:
        reward = sigmoid_reward(relative_distance)
    except:
        reward = -0.5
    
    # 确保奖励在合理范围内
    return max(min(reward, 1.0), -0.5)

            

def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    统一 reward 范围至 [-1, 1]，支持三类任务（排序、选择、时间）
    """
    import json
    import re
    from datetime import datetime
    from dateutil.parser import parse
    # -------------------------

    # -------------------------
    # ① 解析 response json
    # -------------------------
    try:
        # 清理可能的markdown代码块标记
        cleaned_str = solution_str.strip()
        
        # 移除markdown代码块标记
        if cleaned_str.startswith('```'):
            # 找到第一个和最后一个```的位置
            start_idx = cleaned_str.find('\n', 3) + 1  # 跳过```json\n
            end_idx = cleaned_str.rfind('```')
            if end_idx > start_idx:
                cleaned_str = cleaned_str[start_idx:end_idx].strip()
        
        # 尝试解析JSON
        response_json = json.loads(cleaned_str)
    except Exception as e:
        print("Error input:", cleaned_str)
        return -1.0  # 无法解析为合法 JSON → 最低惩罚
    
    try:
        pred_answer = response_json.get("answer", None)

        # -------------------------
        # ② 解析 ground truth
        gt_answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth

        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        qtype = extra_info.get("ability", "default")
        all_sessions = extra_info.get("all_sessions", []) if extra_info else []
        prompt = extra_info.get("question", [])
        question_context = extract_question_content(prompt)
        
        def calculate_answer_reward(pred, gold, task_type):
            pred_str = str(pred).strip() if pred is not None else ""
            gold_str = str(gold).strip() if gold is not None else ""
            task_type = task_type.lower()
            #print("-----------------------------task-type---------------------------------")
            #print(task_type)
            #print("-----------------------------------------------------------------------")
            if qtype == "localization":
                pred_str = normalize_date(pred_str)
                gold_str = normalize_date(gold_str)
            try:
                if task_type == "computation":
                    try:
                        metrics = calculate_computation_metrics(pred_str, gold_str)
                        #print("{\n")
                        #print(pred_str)
                        #print(gold_str)
                        #print(metrics["exact_match"])
                        #print("\n}")
                        return metrics["exact_match"]
                    except Exception as e:
                        #print(e)
                        #print(-0.5)
                        return -0.5
                elif task_type == "timeline":
                    try:
                        reward = calculate_timeline_metrics(pred_str, gold_str)
                        #print("{\n")
                        #print(pred_str)
                        #print(gold_str)
                        #print(metrics["TP"])
                        #print("\n}")
                        return reward
                    except Exception as e:
                        #print(e)
                        #print(-0.5)
                        return -0.5
                elif task_type == "localization":
                    try:
                        pred_ans = normalize_date(pred_str)
                        gold_ans = normalize_date(gold_str)
                        metrics = calculate_localization_metrics(pred_ans, gold_ans)
                        #print("{\n")
                        #print(pred_ans)
                        #print(gold_ans)
                        #print(metrics["exact_match"])
                        #print("\n}")
                        return metrics["exact_match"]
                    except Exception as e:
                        #print(e)
                        #print(-0.5)
                        return -0.5
                else:
                    pred_set = extract_options(pred_str)
                    gold_set = extract_options(gold_str)
                    metrics = calculate_metrics(pred_set, gold_set)
                    #print("{\n")
                    #print(pred_set)
                    #print(gold_set)
                    #print(metrics["exact_match"])
                    #print("\n}")
                    return metrics["exact_match"]
            except Exception as e:
                print(f"Error in calculate_answer_reward for {task_type}: {e}")
                return -0.5
        
        accuracy_reward = calculate_answer_reward(pred_answer, gt_answer, qtype)
        # 处理不同的字段名和格式

        reward = accuracy_reward
        return reward
    except Exception as e:
        print(e)
        return 0.0
