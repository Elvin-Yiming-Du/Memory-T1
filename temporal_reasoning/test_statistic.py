# import json
# import csv
# import torch
# import torch_npu
# from transformers import AutoTokenizer

# # 检查是否有NPU可用
# device = "npu" if torch.npu.is_available() else "cpu"

# # 加载Qwen的tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "/home/ma-user/work/ydu/models/Qwen/Qwen2.5_7B_Instruct",
#     trust_remote_code=True,
#     device_map=device
# )

# def count_tokens(text):
#     # 编码文本并返回token数量
#     return len(tokenizer.encode(text))

# def load_time_contexts():
#     """加载带时间标注的对话文件"""
#     context_file = "/home/ma-user/work/ydu/data/time_range/time_dial_labeled_contexts.json"
#     with open(context_file, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def format_session_context(session_id, session: dict) -> str:
#     """将会话格式化为文本"""
#     context = f"{session.get('session_date')}: <{str(session_id)}>, "
#     utterances = []
#     for utterance in session.get('content', []):
#         utterance_text = f"{utterance.get('speaker')} : {utterance.get('utterance_date')} : {utterance.get('utterance')}"
#         utterances.append(utterance_text)
#     context += "\n".join(utterances)
#     return context

# def main():
#     # ====== 路径配置 ======
#     time_contexts = load_time_contexts()
#     question_file = "/home/ma-user/work/ydu/data/time_range/time_dial_context_list_with_evidence_200_v2.json"
#     output_csv = "/home/ma-user/work/ydu/data/time_range/prompt_length_analysis.csv"
    
#     # ====== 加载数据 ======
#     print("Loading questions...")
#     with open(question_file, 'r', encoding='utf-8') as f:
#         questions = json.load(f)

#     # ====== 定义长度区间 ======
#     length_ranges = [
#         (0, 8000), (8000, 16000), (16000, 32000), (32000, 64000),
#         (64000, 128000)
#     ]
    
#     # 初始化统计结果
#     range_counts = {f"{start//1000}k-{end//1000}k": 0 for start, end in length_ranges}
#     question_length_data = []
    
#     # ====== 统计处理 ======
#     print("Processing questions and counting tokens...")
    
#     for i, question in enumerate(questions):
#         print(f"Processing question {i+1}/{len(questions)}")
        
#         session_context = question.get('Context', '')
#         if session_context not in time_contexts:
#             continue
            
#         evidence_sessions = time_contexts[session_context]
#         full_sessions = []
        
#         for session_id, session in reversed(list(evidence_sessions.items())):
#             formatted_session_content = format_session_context(session_id, session)
#             full_sessions.append(formatted_session_content)
        
#         # 构建prompt并计算token数量
#         prompt_text = "\n".join(full_sessions)
#         token_count = count_tokens(prompt_text)
        
#         # 确定长度范围类别
#         length_category = "128k+"
#         for start, end in length_ranges:
#             if start <= token_count < end:
#                 length_category = f"{start//1000}k-{end//1000}k"
#                 range_counts[length_category] += 1
#                 break
        
#         # 保存每个问题的详细信息
#         question_data = {
#             'question': question['Question'],
#             'context': session_context,
#             'token_count': token_count,
#             'length_category': length_category
#         }
#         question_length_data.append(question_data)
    
#     # ====== 输出统计结果 ======
#     print("\n=== Prompt长度分布统计 ===")
#     print("长度区间\t\t问题数量")
#     print("-" * 30)
    
#     total_questions = 0
#     for range_key in sorted(range_counts.keys(), key=lambda x: int(x.split('-')[0][:-1])):
#         count = range_counts[range_key]
#         total_questions += count
#         print(f"{range_key}\t\t{count}")
    
#     print("-" * 30)
#     print(f"总计\t\t{total_questions}")
    
#     # ====== 保存到CSV文件 ======
#     print(f"\nSaving results to {output_csv}...")
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['question', 'context', 'token_count', 'length_category']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         writer.writeheader()
#         for data in question_length_data:
#             writer.writerow(data)
    
#     print(f"CSV file saved successfully with {len(question_length_data)} records.")

# if __name__ == "__main__":
#     main()

import json
import csv
import torch
import torch_npu
from transformers import AutoTokenizer

# 检查是否有NPU可用
device = "npu" if torch.npu.is_available() else "cpu"

# 加载Qwen的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ma-user/work/ydu/models/Qwen/Qwen2.5_7B_Instruct",
    trust_remote_code=True,
    device_map=device
)

def count_tokens(text):
    # 编码文本并返回token数量
    return len(tokenizer.encode(text))

def load_time_contexts():
    """加载带时间标注的对话文件"""
    context_file = "/home/ma-user/work/ydu/data/time_range/time_dial_labeled_contexts.json"
    with open(context_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_session_context(session_id, session: dict) -> str:
    """将会话格式化为文本"""
    context = f"{session.get('session_date')}: <{str(session_id)}>, "
    utterances = []
    for utterance in session.get('content', []):
        utterance_text = f"{utterance.get('speaker')} : {utterance.get('utterance_date')} : {utterance.get('utterance')}"
        utterances.append(utterance_text)
    context += "\n".join(utterances)
    return context

def main():
    # ====== 路径配置 ======
    time_contexts = load_time_contexts()
    question_file = "/home/ma-user/work/ydu/data/time_range/time_dial_context_list_with_evidence_200_v2.json"
    output_csv = "/home/ma-user/work/ydu/data/time_range/prompt_length_analysis.csv"
    
    # ====== 加载数据 ======
    print("Loading questions...")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # ====== 定义长度区间 ======
    length_ranges = [
        (0, 8000), (8000, 16000), (16000, 32000), (32000, 64000),
        (64000, 128000)
    ]
    
    # 初始化统计结果
    range_counts = {f"{start//1000}k-{end//1000}k": 0 for start, end in length_ranges}
    question_length_data = []
    
    # ====== 统计处理 ======
    print("Processing questions and counting tokens...")
    
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}")
        
        session_context = question.get('Context', '')
        if session_context not in time_contexts:
            continue
            
        evidence_sessions = time_contexts[session_context]
        full_sessions = []
        
        for session_id, session in reversed(list(evidence_sessions.items())):
            formatted_session_content = format_session_context(session_id, session)
            full_sessions.append(formatted_session_content)
        
        # 构建prompt并计算token数量
        prompt_text = "\n".join(full_sessions)
        token_count = count_tokens(prompt_text)
        
        # 确定长度范围类别
        length_category = "128k+"
        for start, end in length_ranges:
            if start <= token_count < end:
                length_category = f"{start//1000}k-{end//1000}k"
                range_counts[length_category] += 1
                break
        
        # 保存每个问题的详细信息
        question_data = {
            'question': question['Question'],
            'token_count': token_count,
            'length_category': length_category
        }
        question_length_data.append(question_data)
    
    # ====== 输出统计结果 ======
    print("\n=== Prompt长度分布统计 ===")
    print("长度区间\t\t问题数量")
    print("-" * 30)
    
    total_questions = 0
    for range_key in sorted(range_counts.keys(), key=lambda x: int(x.split('-')[0][:-1])):
        count = range_counts[range_key]
        total_questions += count
        print(f"{range_key}\t\t{count}")
    
    print("-" * 30)
    print(f"总计\t\t{total_questions}")
    
    # ====== 保存到CSV文件 ======
    print(f"\nSaving results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'token_count', 'length_category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in question_length_data:
            writer.writerow(data)
    
    print(f"CSV file saved successfully with {len(question_length_data)} records.")

if __name__ == "__main__":
    main()