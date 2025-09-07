import json
import os
from typing import Dict, List, Any
from openai import OpenAI
import re
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm
import tiktoken
import time
from datetime import datetime
import torch
import torch_npu  # 导入NPU支持库
from transformers import AutoTokenizer

# 检查是否有NPU可用
device = "npu" if torch.npu.is_available() else "cpu"
# 加载Qwen的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ma-user/work/ydu/models/Qwen/Qwen2.5_3B_Instruct",
    trust_remote_code=True,
    device_map=device
)
with open("/home/ma-user/work/ydu/data/time_range/question_time_range_annotated_0717.json", "r", encoding="utf-8") as f:
    QUESTION_TIME_RANGE_MAP = json.load(f)

def extract_time_range_from_question(question_id):
    # 直接查表
    item = QUESTION_TIME_RANGE_MAP.get(str(question_id))
    
    if item and "question_time_range" in item:
        time_range = item["question_time_range"]
        start_time = "1900-01-01" if time_range["start"] == "unknown" else time_range["start"]
        end_time = "2025-09-01" if time_range["end"] == "unknown" else time_range["end"]
        return {
            "start": datetime.fromisoformat(start_time).date(),
            "end": datetime.fromisoformat(end_time or datetime.now().isoformat()).date()
        }
    else:
        # 没找到时返回默认的date类型，end为当前日期
        return {
            "start": datetime.strptime("1900-01-01", "%Y-%m-%d").date(),
            "end": date.today()
        }


def count_tokens(text):
    # 编码文本并返回token数量
    return len(tokenizer.encode(text))

def load_time_contexts():
    """加载带时间标注的对话文件"""
    context_file = "/home/ma-user/work/ydu/data/time_range/time_dial_labeled_contexts.json"
    with open(context_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_session_context(session_id, session: Dict) -> str:
    """将会话格式化为文本"""
    context = f"{session.get('session_date')}: <session_{str(session_id)}>, "
    utterances = []
    for utterance in session.get('content', []):
        utterance_text = f"{utterance.get('speaker')} : {utterance.get('utterance_date')} : {utterance.get('utterance')}"
        utterances.append(utterance_text)
    context += "\n".join(utterances)
    return context

def format_session(s):
    lines = []
    utterances = " ".join(u['speaker'] + ": " + u['utterance_time'] + ": " + u['text'] + "\n" for u in s['utterances'])
    lines.append(f"{s['session_time']}: <{s['session_id']}> \n {utterances}")
    return "\n".join(lines)


def format_session_for_prompt(session_id, session):
    session_time = session.get('session_date', '')
    lines = []
    for utt in session.get('content', []):
        lines.append(f"{utt.get('utterance_time', '')}: {session_id}: {utt.get('speaker', '')}: {utt.get('content', '')}")
    return f"{session_time}: {session_id}:\n" + "\n".join(lines)

def bm25_retriever(query: str, sessions: Dict, top_k: int = 10) -> List[str]:
    """
    使用BM25检索相关会话
    
    Args:
        query: 查询文本
        sessions: 会话数据
        top_k: 返回的会话数量
    
    Returns:
        相关会话ID列表
    """
    # 准备文档和ID
    documents = []
    session_ids = []
    
    for session_id, session in sessions.items():
        # 格式化会话内容
        doc = format_session_context(session_id, session)
        documents.append(doc)
        session_ids.append(session_id)
    
    # 创建BM25索引
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 对查询进行检索
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    # 获取top-k结果
    top_indices = np.argsort(scores)[-top_k:][::-1]
    retrieved_ids = [session_ids[i] for i in top_indices]
    
    return retrieved_ids

def parse_gpt_session_output(json_str: str) -> Dict[str, Any]:
    try:
        # 清理输入字符串
        json_str = json_str.strip()
        
        # 移除可能的 markdown 代码块标记
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        # 清理字符串并确保是有效的 JSON
        json_str = json_str.strip()
        
        # 尝试解析 JSON
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            # 如果解析失败，尝试修复常见的 JSON 格式问题
            # 1. 移除多余的逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            # 2. 确保键值对格式正确
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            result = json.loads(json_str)

        if not isinstance(result, dict):
            raise ValueError("Parsed result is not a JSON object")

        if "answer" not in result:
            raise KeyError("Missing required key: 'answer'")

        return result

    except Exception as e:
        print(f"[parse_gpt_session_output] Error parsing JSON: {e}")
        print(f"Problematic JSON string: {json_str}")
        return {"error": str(e)}

def build_prompt(question, dialogue_history, now=None):
    response_format = f"""You are a memory-aware reasoning assistant. Your task is to answer temporal questions based on multi-turn dialogue history. Carefully analyze the provided context, reason about time and events, and respond strictly in JSON format.
        Your response  must be a valid JSON object with the following structure:
        ```json
        {{
          "selected_memory": ["session_X", "session_Y"],
          "answer": "X"
        }}
        ```

        ## Answer Format Rules (By Type)
        1. Single choice: Uppercase letter (e.g., "A")
        2. Multiple choice: Space-separated uppercase letters in order (e.g., "A C E")
        3. Time: "HH:MM:SS [am/pm], FullMonthName DD, YYYY", Example: "02:30:00 pm, March 22, 2024"
        4. For sequence answers, enclose each digit in parentheses and list them in order, e.g.,  (e.g., "(1)(3)(2)(4)(5)(6)(8)(7)")

        ## Input Format
        You are the given dialogue history:
        <previous_memory>{dialogue_history}</previous_memory>

        You current task if answer the following question
        <question>
        {str(now)}：
        Question: {question}
        </question>

        Please respond only with a valid JSON object as shown above. 

        ##For example
        ```json
        {{
          "selected_memory": ["session_1", "session_7"],
          "answer": "A"
        }}
        ```

    """
    return response_format
    

def infer_with_local_model(prompt, tokenizer, model, temperature=0.1, top_p=0.95, max_new_tokens=128, stop=None):
    prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # positional argument
            add_generation_prompt=True,
            tokenize=False)
 
    
    inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=15500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
     # 记录 prompt token 长度
    input_token_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    # 只 decode 新生成部分
    generated_ids = outputs[0][input_token_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(answer)
    return answer
    
def parse_model_output(output):
    match = re.search(r'\{[\s\S]*\}', output)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}

def load_model_and_tokenizer(base_model_path, lora_adapter_path):
    """加载模型和分词器"""
    # === 3. 加载 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

      # 指定目标设备
    target_device = "npu:0"  # 或者 "cuda:0" 根据你的硬件
    
    # === 4. 加载 base Qwen 模型 ===
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True  # Qwen 必须设置为 True
    ).to(target_device)
    
    if lora_adapter_path!="":
        # === 5. 加载 LoRA adapter ===
        model = PeftModel.from_pretrained(
            model,
            lora_adapter_path,
            device_map=target_device,
            torch_dtype=torch.float16,
        )

    return tokenizer, model

def main():
    # ====== 路径配置（请根据实际情况修改） ======
    time_contexts = load_time_contexts()
    question_file = "/home/ma-user/work/ydu/data/time_range/time_dial_context_list_with_evidence_200_v2.json"  # 问题列表，每个元素至少有Question字段
    output_file = "/home/ma-user/work/ydu/temporal_reasoning/results/time_range_rl_bm25_qwen2.5_3B_acc-fmt-tc_results.json"
    max_prompt_length = 15500
    base_model_path = "/home/ma-user/work/ydu/models/MyTrain/Qwen2.5-3B-Instruct-Time-RL-acc-fmt-tc-0824"  # 基础模型路径
    lora_adapter_path = ""

    tokenizer, model = load_model_and_tokenizer(base_model_path, lora_adapter_path)

    # ====== 加载数据 ======
    print("Loading sessions and questions...")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # ====== 主流程 ======
    results = []
    recall_at_5 = 0
    total = 0
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}")
        query = question["Question"]
        gold_sessions = question.get('Evidence')
        # 1. BM25检索top5
        session_context = question.get('Context', '')
        evidence_sessions = time_contexts[session_context]
        time_range =extract_time_range_from_question(question.get("Question ID"))
        # print(time_range)
        candidate_sessions = {}
        for session_id, session in evidence_sessions.items():
            session_time = session.get('session_date', '')
            session_date = datetime.fromisoformat(session_time).date()
            if time_range["start"] <= session_date <= time_range["end"]:
                candidate_sessions[session_id] = session
        
        if len(candidate_sessions) == 0:
            candidate_sessions = evidence_sessions
        
        candidate_ids = []
        for session_id in candidate_sessions.keys():
            candidate_ids.append(session_id)

        # Recall@5: gold_sessions为list，统计每个gold session是否在top5_ids中
        retrieved_sessions = []
        load_sessions_length = 0

        for session_id in candidate_ids:
            formatted_session_content = format_session_context(session_id, evidence_sessions[session_id])
            load_sessions_length += count_tokens(formatted_session_content)
            if load_sessions_length < max_prompt_length:
                retrieved_sessions.append(formatted_session_content)
            else:
                break

        if gold_sessions:
            for evident_session in gold_sessions:
                if evident_session["session_id"] in candidate_ids:
                    recall_at_5 += 1
            total += len(gold_sessions)
            
        CURRENT_TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 2. 本地模型推理
        prompt = build_prompt(query, "\n".join(retrieved_sessions), CURRENT_TIME)
        answer = infer_with_local_model(prompt, tokenizer, model)
        # print(answer)
        parsed = parse_model_output(answer)
        result = {
            "question_id": question.get("Question ID"),
            "question": query,
            "level": question.get("Level"),
            "task": question.get("Task"),
            "time_constraint_ids": candidate_ids,
            "prompt": prompt,
            "model_output": answer,
            "parsed_output": parsed,
            "gold_answer": question.get("Gold Answer", "")
        }
        results.append(result)
        # 保存中间结果
    
    # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved intermediate results to {output_file}")
    if total > 0:
        print(f"Recall@5: {recall_at_5/total:.3f}")
    else:
        print("No gold session provided, Recall@5 not computed.")
    print(f"\nProcessing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main() 