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
        print(response_json)
        pred_answer = response_json.get("answer", None)

        # -------------------------
        # ② 解析 ground truth
        # -------------------------
        gt_answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth

        # -------------------------
        # ③ 获取 question type
        # -------------------------
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        qtype = extra_info.get("ability", "default")

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
                'Counterfactual', 'Co_temporality', 'Relative_Reasoning', 
                'Duration_Compare', 'Order_Reasoning', 'Order_Compare', 
                'Explicit_Reasoning', 'Extract'
            ]

            # 事件排序类型
            event_order_types = ['Timeline']

            # 时间跨度类型
            time_span_types = ['Computation', 'Localization']

            if ability_type in single_multi_choice_types:
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

            elif ability_type in event_order_types:
                # 处理事件排序答案 - 提取数字序列
                return re.findall(r'\d+', ans)

            elif ability_type in time_span_types:
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
        else:
            accuracy_reward = -1

        # -------------------------
        # ⑥ Compression reward ∈ [-1, 1]
        # -------------------------
        # 处理不同的字段名和格式
        selected_sessions_raw = response_json.get("selected_sessions", response_json.get("selected_memory", []))

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

        all_sessions = extra_info.get("all_sessions", []) if extra_info else []
        total_sessions = max(len(all_sessions), 1)

        compression_ratio = len(selected_sessions) / total_sessions
        selected_compression_reward = 2 * (0.5 - compression_ratio)  # 压缩越多 → 趋近 +1

        # -------------------------
        # ⑦ Selection accuracy ∈ [-1, 1]
        # -------------------------
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
        question_time_range = extra_info.get("question_time_range", {}) if extra_info else {}
        start_time = question_time_range.get("start")
        end_time = question_time_range.get("end")

        def in_time_range(session_id):
            for s in all_sessions:
                if s.get("session_id") == session_id:
                    for utt in s.get("utterances", []):
                        utt_time = utt.get("utterance_time")
                        if utt_time and start_time and end_time:
                            try:
                                utt_dt = datetime.fromisoformat(utt_time)
                                start_dt = datetime.fromisoformat(start_time)
                                end_dt = datetime.fromisoformat(end_time)
                                if not (start_dt <= utt_dt <= end_dt):
                                    return False
                            except:
                                continue
            return True

        if len(selected_sessions) == 0:
            time_consistency = 0.0
        else:
            valid = sum(in_time_range(sid) for sid in selected_sessions)
            if valid == len(selected_sessions):
                time_consistency = 1.0
            elif valid == 0:
                time_consistency = 0.0
            else:
                time_consistency = -1

        # -------------------------
        # ⑨ 合成 reward（平均）
        # -------------------------
        reward = (accuracy_reward + selected_compression_reward + selected_accuracy + time_consistency) / 4

        return reward
    except Exception as e:
        print(e)
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, method="strict", format_score=0.0, score=1.0):
    """
    统一 reward 范围至 [-1, 1]，支持三类任务（排序、选择、时间）
    """
    import json
    import re
    from datetime import datetime
    from dateutil.parser import parse
   
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
        return -1.0  # 无法解析为合法 JSON → 最低惩罚
    
    pred_answer = response_json.get("answer", None)

    # -------------------------
    # ② 解析 ground truth
    # -------------------------
    gt_answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth

    # -------------------------
    # ③ 获取 question type
    # -------------------------
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    qtype = extra_info.get("ability", "default")

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
            'Counterfactual', 'Co_temporality', 'Relative_Reasoning', 
            'Duration_Compare', 'Order_Reasoning', 'Order_Compare', 
            'Explicit_Reasoning', 'Extract'
        ]
        
        # 事件排序类型
        event_order_types = ['Timeline']
        
        # 时间跨度类型
        time_span_types = ['Computation', 'Localization']
        
        if ability_type in single_multi_choice_types:
            # 处理单选择/多选择题答案
            # 首先检查是否是排序格式 (1)(2)(3) 或 (1,2,3)
            if re.match(r'^[\(\d\)\s,]+$', ans):
                # 格式错误：单选题不应该输出排序格式
                return None  # 返回None表示格式错误，将在reward计算中给予惩罚
            else:
                # 按选择题处理
                choices = sorted(set([a.strip() for a in re.split(r'[,\s]+', ans) if a.strip() in ['A', 'B', 'C', 'D']]))
                if not choices:
                    # 如果没有找到有效的选项，也是格式错误
                    return None
                return choices
        
        elif ability_type in event_order_types:
            # 处理事件排序答案 - 提取数字序列
            return re.findall(r'\d+', ans)
        
        elif ability_type in time_span_types:
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
    else:
        accuracy_reward = -1.0

    # -------------------------
    # ⑥ Compression reward ∈ [-1, 1]
    # -------------------------
    # 处理不同的字段名和格式
    selected_sessions_raw = response_json.get("selected_sessions", response_json.get("selected_memory", []))
    
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
    
    all_sessions = extra_info.get("all_sessions", []) if extra_info else []
    total_sessions = max(len(all_sessions), 1)

    compression_ratio = len(selected_sessions) / total_sessions
    selected_compression_reward = 2 * (0.5 - compression_ratio)  # 压缩越多 → 趋近 +1

    # -------------------------
    # ⑦ Selection accuracy ∈ [-1, 1]
    # -------------------------
    gold_selected_sessions = extra_info.get("gold_selected_sessions", []) if extra_info else []
    selected_set = set(selected_sessions)
    gold_set = set(gold_selected_sessions)

    if selected_set == gold_set:
        selected_accuracy = 1.0
    elif selected_set & gold_set:
        selected_accuracy = 0.5
    else:
        selected_accuracy = -1.0

    # -------------------------
    # ⑧ Time consistency ∈ [-1, 1]
    # -------------------------
    question_time_range = extra_info.get("question_time_range", {}) if extra_info else {}
    start_time = question_time_range.get("start")
    end_time = question_time_range.get("end")

    def in_time_range(session_id):
        for s in all_sessions:
            if s.get("session_id") == session_id:
                for utt in s.get("utterances", []):
                    utt_time = utt.get("utterance_time")
                    if utt_time and start_time and end_time:
                        try:
                            utt_dt = datetime.fromisoformat(utt_time)
                            start_dt = datetime.fromisoformat(start_time)
                            end_dt = datetime.fromisoformat(end_time)
                            if not (start_dt <= utt_dt <= end_dt):
                                return False
                        except: 
                            continue
        return True

    if len(selected_sessions) == 0:
        time_consistency = 0.0
    else:
        valid = sum(in_time_range(sid) for sid in selected_sessions)
        if valid == len(selected_sessions):
            time_consistency = 1.0
        elif valid == 0:
            time_consistency = -1.0
        else:
            time_consistency = 0.0

    # -------------------------
    # ⑨ 合成 reward（平均）
    # -------------------------
    reward = (accuracy_reward + selected_compression_reward + selected_accuracy + time_consistency) / 4
    
    return reward
