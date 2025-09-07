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
        # -------------------------
        # ④ 标准化答案
        # -------------------------
        def normalize_answer(ans, ability_type):
            """
            根据ability类型标准化答案格式

            Args:
                ans: 原始答案
                ability_type: ability类型，如 'Counterfactual', 'Timeline', 'Computation' 等
            """
            if ans is None:
                return None

            ans = str(ans).strip().upper()

            # 单选择/多选择题类型
            single_multi_choice_types = [
                'counterfactual', 'co_temporality', 'relative_reasoning', 
                'duration_Compare', 'order_Reasoning', 'order_compare', 
                'explicit_Reasoning', 'extract'
            ]

            # 事件排序类型
            event_order_types = ['timeline']

            # 时间跨度类型
            time_span_types = ['computation', 'localization']

            if ability_type.lower() in single_multi_choice_types:
                # 处理单选择/多选择题答案
                # 首先检查是否是排序格式 (1)(2)(3) 或 (1,2,3)
                if re.match(r'^[\(\d\)\s,]+$', ans):
                    # 格式错误：单选题不应该输出排序格式
                    # print(f"格式错误: 单选题 {ability_type} 输出了排序格式 '{ans}'")
                    return None  # 返回None表示格式错误，将在reward计算中给予惩罚
                else:
                    # 按选择题处理
                    choices = sorted(set([a.strip() for a in re.split(r'[,\s]+', ans) if a.strip() in ['A', 'B', 'C', 'D']]))
                    if not choices:
                        # 如果没有找到有效的选项，也是格式错误
                        # print(f"格式错误: 单选题 {ability_type} 没有找到有效选项 '{ans}'")
                        return None
                    return choices

            elif ability_type.lower() in event_order_types:
                # 处理事件排序答案 - 提取数字序列
                return re.findall(r'\d+', ans)

            elif ability_type.lower() in time_span_types:
                # 处理时间跨度答案 - 尝试解析日期
                try:
                    return parse(ans).date().isoformat()
                except:
                    return ans

            else:
                # 默认处理
                return ans

        norm_gt = normalize_answer(gt_answer, qtype)
        norm_pred = normalize_answer(pred_answer, qtype)

        # 检查答案格式
        if norm_pred is None:
            accuracy_reward = -0.5
        elif norm_pred == norm_gt:
            accuracy_reward = 1.0
        elif qtype.lower() in ['timeline']:
            # 时间线顺序匹配：按顺序匹配的比例给分
            if not norm_gt or not norm_pred:
                accuracy_reward = -0.5
            else:
                # 计算顺序匹配的数量
                correct_order_count = 0
                min_length = min(len(norm_gt), len(norm_pred))

                for i in range(min_length):
                    if norm_gt[i] == norm_pred[i]:
                        correct_order_count += 1

                # 按比例给分，最高1分
                accuracy_reward = correct_order_count / len(norm_gt) if len(norm_gt) > 0 else 0

        elif qtype.lower() in [
            'counterfactual', 'co_temporality', 'relative_reasoning', 
            'duration_compare', 'order_reasoning', 'order_compare', 
            'explicit_reasoning', 'extract'
        ]:
            # 选择类型：集合交叉不为空则给0.5
            if not norm_gt or not norm_pred:
                accuracy_reward = -0.5
            else:
                # 转换为集合进行比较
                gt_set = set(norm_gt)
                pred_set = set(norm_pred)

                if gt_set & pred_set:  # 集合交集不为空
                    accuracy_reward = 0.5
                else:
                    accuracy_reward = -1
        else:
            accuracy_reward = -1

        # 处理不同的字段名和格式
        selected_sessions_raw =response_json.get("selected_memory", [])

        # 清理session ID格式，移除尖括号，并验证格式
        selected_sessions = []
        for session in selected_sessions_raw:
            if isinstance(session, str):
                # 移除尖括号
                cleaned_session = session.strip('<>')
                # 验证session格式：必须包含session_数字的格式
                match = re.search(r'(session_\d+)', cleaned_session)
                if match:
                    selected_sessions.append(match.group(1))  # 只添加匹配到的session_id
            else:
                session_str = str(session)
                match = re.search(r'(session_\d+)', session_str)
                if match:
                    selected_sessions.append(match.group(1))  # 只添加匹配到的session_id

        gold_selected_sessions = extra_info.get("gold_selected_sessions", []) if extra_info else []
        selected_set = set(selected_sessions)
        gold_set = set(gold_selected_sessions)

        if selected_set == gold_set:
            selected_accuracy = 1.0
        elif selected_set & gold_set:
            selected_accuracy = 0.5
        else:
            selected_accuracy = -1

        # -------------------------
        # ⑧ Time consistency ∈ [-1, 1]
        # -------------------------
        question_time_range_str = extra_info.get("question_time_range", "") if extra_info else ""
        question_time_range = parse_stored_time_range(question_time_range_str)
        start_time = question_time_range.get("start")
        #1 means start time, 0 means end time
        start_time = parse_timestamp_with_default(start_time, 1)
        end_time = question_time_range.get("end")
        end_time = parse_timestamp_with_default(end_time, 0)


        def in_time_range(session_id, start_time, end_time):
            time_range_reward = 0.0
            time_density_reward = 0.0
            valid_event_counter = 0
            accumulate_density = 0.0
            for s in all_sessions:
                if s.get("session_id") == session_id:

                    dialog_session_time = s.get("session_time","")

                    dialog_session_time = parse_timestamp_with_default(dialog_session_time, 1)
                    time_range_reward = calculate_session_time_reward(dialog_session_time, start_time, end_time)

                    for utt in s.get("utterances", []):
                        try:
                            events = utt.get("events")
                            for event in events:
                                if event["event_time"][0] == "unknown" and event["event_time"][1] == "unknown":
                                    continue
                                else:
                                    text_similary_score = simple_text_match(event["event_summary"], question_context)
                                    if text_similary_score < 0.2:
                                        continue
                                    event_start_time = parse_timestamp_with_default(event["event_time"][0],1)
                                    event_end_time = parse_timestamp_with_default(event["event_time"][1],0)
                                    accumulate_density += text_similary_score * check_time_range_relation(start_time, end_time, event_start_time, event_end_time)
                                    valid_event_counter+=1
                        except Exception as e:
                            print(e)
                            continue
            if valid_event_counter == 0:
                time_density_reward = 0
            else:
                time_density_reward = accumulate_density/valid_event_counter

            return (time_range_reward + time_density_reward)/2

        if len(selected_sessions) == 0:
            time_consistency = 0.0
        else:
            time_rewards = sum(in_time_range(sid, start_time, end_time) for sid in selected_sessions)
            time_consistency = time_rewards/len(selected_sessions)

        # -------------------------
        # ⑨ 合成 reward（平均）
        # -------------------------
        reward = (accuracy_reward + selected_accuracy + time_consistency) / 3

        return reward
    except Exception as e:
        print(e)
        return 0.0
