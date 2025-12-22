# reward_fn.py
# 用于 VERL 中自定义 reward_function.path 的评估文件

def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义 VERL reward 函数接口
    参数：
    - data_source: str, 表示数据来源，如 "temporal_reasoning"
    - response: str, 模型生成的回答
    - ground_truth: str, 正确答案
    - extra_info: dict or None, 额外信息（如题型、难度等）
    返回：float, 作为强化学习训练时的 reward 分数
    """
    import json
    from datetime import datetime


    print(ground_truth)
    print("*"*100)
    print(solution_str)
    print("*"*100)
    print(data_source)
    print("*"*100)
    response = solution_str.strip()

    # 解析模型输出
    try:
        response_json = json.loads(response)
    except Exception:
        return -1.0  # 格式错误直接给负分

    # 1. accuracy_reward
    gt_answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth
    pred_answer = response_json.get("answer")
    accuracy_reward = 1.0 if str(pred_answer).strip().upper() == str(gt_answer).strip().upper() else 0.0

    # 2. selected_compression_ratio
    selected_sessions = response_json.get("selected_sessions", [])
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    all_sessions = extra_info.get("all_sessions", []) if extra_info else []
    total_sessions = len(all_sessions) if all_sessions else 1
    selected_compression_ratio = 1.0 - (len(selected_sessions) / total_sessions) if total_sessions > 0 else 0.0

    # 3. selected_accuracy
    gold_selected_sessions = extra_info.get("gold_selected_sessions", []) if extra_info else []
    if set(selected_sessions) == set(gold_selected_sessions):
        selected_accuracy = 1.0
    elif set(selected_sessions) & set(gold_selected_sessions):
        selected_accuracy = 0.5
    else:
        selected_accuracy = 0.0

    # 4. time_consistency
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
                        except Exception:
                            continue
        return True
    time_consistency = 1.0 if all(in_time_range(sid) for sid in selected_sessions) else 0.0

    # 总 reward
    reward = accuracy_reward + selected_compression_ratio + selected_accuracy + time_consistency
    return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None ,method="strict", format_score=0.0, score=1.0):
    """
    自定义 VERL reward 函数接口
    参数：
    - data_source: str, 表示数据来源，如 "temporal_reasoning"
    - response: str, 模型生成的回答
    - ground_truth: str, 正确答案
    - extra_info: dict or None, 额外信息（如题型、难度等）
    返回：float, 作为强化学习训练时的 reward 分数
    """
    import json
    from datetime import datetime


    print(ground_truth)
    print("*"*100)
    print(solution_str)
    print("*"*100)
    response = solution_str.strip()
    print(data_source)
    print("*"*100)

    # 解析模型输出
    try:
        response_json = json.loads(response)
    except Exception:
        return -1.0  # 格式错误直接给负分

    # 1. accuracy_reward
    gt_answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth
    pred_answer = response_json.get("answer")
    accuracy_reward = 1.0 if str(pred_answer).strip().upper() == str(gt_answer).strip().upper() else 0.0

    # 2. selected_compression_ratio
    selected_sessions = response_json.get("selected_sessions", [])
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    all_sessions = extra_info.get("all_sessions", []) if extra_info else []
    total_sessions = len(all_sessions) if all_sessions else 1
    selected_compression_ratio = 1.0 - (len(selected_sessions) / total_sessions) if total_sessions > 0 else 0.0

    # 3. selected_accuracy
    gold_selected_sessions = extra_info.get("gold_selected_sessions", []) if extra_info else []
    if set(selected_sessions) == set(gold_selected_sessions):
        selected_accuracy = 1.0
    elif set(selected_sessions) & set(gold_selected_sessions):
        selected_accuracy = 0.5
    else:
        selected_accuracy = 0.0

    # 4. time_consistency
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
                        except Exception:
                            continue
        return True
    time_consistency = 1.0 if all(in_time_range(sid) for sid in selected_sessions) else 0.0

    # 总 reward
    reward = accuracy_reward + selected_compression_ratio + selected_accuracy + time_consistency
    return reward