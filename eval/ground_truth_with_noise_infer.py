# 该评测只是评测过程的一部分，第一部分的记忆选择在time_qa_eval.py的脚本中完成。

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
import torch
import torch_npu  # 导入NPU支持库
from transformers import AutoTokenizer
import random  # 添加random模块
import random
import math

# 检查是否有NPU可用
device = "npu" if torch.npu.is_available() else "cpu"
# 加载Qwen的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ma-user/work/ydu/models/Qwen/Qwen2.5_3B_Instruct",
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

def format_session_context(session_id, session: Dict) -> str:
    """将会话格式化为文本"""
    context = f"{session.get('session_date')}: <{str(session_id)}>, "
    utterances = []
    for utterance in session.get('content', []):
        utterance_text = f"{utterance.get('speaker')} : {utterance.get('utterance_date')} : {utterance.get('utterance')}"
        utterances.append(utterance_text)
    context += "\n".join(utterances)
    return context

def format_session_for_prompt(session_id, session):
    session_time = session.get('session_date', '')
    lines = []
    for utt in session.get('content', []):
        lines.append(f"{utt.get('utterance_time', '')}: {session_id}: {utt.get('speaker', '')}: {utt.get('content', '')}")
    return f"{session_time}: {session_id}:\n" + "\n".join(lines)

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

def build_prompt(question, dialogue_sessions, now):
    response_format =  f"""You are presented with a temporal question and a previous memory, please answer the question with the correct format. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

### Output Requirements:
1. **Answer Format** (MUST match one of these exactly):
   - Single choice: `A` | `B` | etc. (uppercase, no quotes)
   - Multiple choice: `A B C` (space-separated uppercase)
   - Time: `HH:MM:SS [am|pm], Month Day, Year` (e.g., `10:45:41 pm, January 15, 2024`)
   - Sequence: `(1)(3)(2)(4)(5)(6)(7)(8)` (numbered parentheses, no spaces)

<previous_memory>{dialogue_sessions}</previous_memory>

<question>
Time: {str(now)}
Question: {question}
</question>

\n\nRemember to put your answer on its own line after \"Answer:\"
"""
    return response_format

def infer_with_local_model(prompt, tokenizer, model, temperature=0.1, top_p=0.95, max_new_tokens=128, stop=None):

    prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # positional argument
            add_generation_prompt=True,
            tokenize=False
        )
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
    # match = re.search(r'\{[\s\S]*\}', output)
    res = output.replace("Answer: ","")
    return res

def load_model_and_tokenizer(base_model_path, lora_model_path=""):
    """加载LoRA模型和分词器 - 保守版本"""
    from peft import PeftModel
    
    # === 加载 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 指定目标设备
    target_device = "npu:0"  # 或者 "cuda:0" 根据你的硬件
    
    # === 加载 base Qwen 模型到指定设备 ===
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=None,  # 禁用自动设备映射
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(target_device)
    
    if lora_model_path != "":
        # === 加载 LoRA 适配器到相同设备 ===
        model = PeftModel.from_pretrained(
            model, 
            lora_model_path,
            device_map={"": target_device}  # 明确指定设备
        )
        print(f"LoRA adapter loaded to device: {target_device}")
    
    return tokenizer, model

def main():
    # ====== 路径配置（请根据实际情况修改） ======
    time_contexts = load_time_contexts()
    question_file = "/home/ma-user/work/ydu/data/time_range/time_dial_context_list_with_evidence_200_v2.json"  # 问题列表，每个元素至少有Question字段
    output_file = "/home/ma-user/work/ydu/temporal_reasoning/results/select_then_infer/gold_with_noise_acc_results_0826_03.json"
    max_prompt_length = 15500
    base_model_path = "/home/ma-user/work/ydu/models/MyTrain/Qwen2.5-3B-Instruct-Time-RL-acc-0823"
    lora_model_path=""
    tokenizer, model = load_model_and_tokenizer(base_model_path, lora_model_path)
    

    # 设置随机种子以确保结果可复现
    random.seed(42)

    # ====== 加载数据 ======
    print("Loading sessions and questions...")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # ====== 主流程 ======
    results = []
    total = 0
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}")
        query = question["Question"]
        gold_sessions = question.get('Evidence')
        # 1. BM25检索top5
        session_context = question.get('Context', '')
        evidence_sessions = time_contexts[session_context]
        gold_session_ids = []
        print(f"\nProcessing question {i+1}/{len(questions)}")

        # 获取金标准ID列表
        gold_sessions = question.get('Evidence', [])
        if gold_sessions:
            for evident_session in gold_sessions:
                gold_session_ids.append(evident_session["session_id"])
        
        # 获取当前上下文中的所有会话ID
        all_session_ids = list(evidence_sessions.keys())
        
        # 随机选择50%的金标准会话替换为噪声会话
        target_ratio = 0.3
        k = len(gold_session_ids)
        r = k * target_ratio
        num_to_replace = math.floor(r) + (1 if random.random() < (r-math.floor(r)) else 0)
        
        sessions_to_replace = random.sample(gold_session_ids, num_to_replace)
        
        # 创建用于构建上下文的会话ID列表
        selected_session_ids = []
        for session_id in gold_session_ids:
            if session_id in sessions_to_replace:
                # 从非金标准会话中随机选择一个作为噪声
                non_gold_sessions = [s for s in all_session_ids if s not in gold_session_ids]
                if non_gold_sessions:  # 确保有非金标准会话可用
                    noise_session = random.choice(non_gold_sessions)
                    selected_session_ids.append(noise_session)
                else:
                    # 如果没有非金标准会话，保留原会话
                    selected_session_ids.append(session_id)
            else:
                selected_session_ids.append(session_id)
        
        load_sessions_length = 0
        selected_sessions_context = []
        for session_id in selected_session_ids:
            formatted_session_content = format_session_context(session_id, evidence_sessions[session_id])
            load_sessions_length += count_tokens(formatted_session_content)
            if load_sessions_length < max_prompt_length:
                selected_sessions_context.append(formatted_session_content)
            else:
                break

        CURRENT_TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 2. 本地模型推理
        prompt = build_prompt(query, "\n".join(selected_sessions_context), CURRENT_TIME)
        answer = infer_with_local_model(prompt, tokenizer, model)
        parsed = parse_model_output(answer)
        print("parsed_result:",parsed)
        print("gold_answer:",question.get("Gold Answer", ""))

        # 记录哪些会话被替换了
        replaced_sessions = {}
        for i, session_id in enumerate(gold_session_ids):
            if session_id in sessions_to_replace:
                replaced_sessions[session_id] = selected_session_ids[i] if i < len(selected_session_ids) else session_id

        result = {
            "question_id": question.get("Question ID"),
            "question": query,
            "level": question.get("Level"),
            "task": question.get("Task"),
            "prompt": prompt,
            "model_output": answer,
            "parsed_output": parsed,
            "gold_answer": question.get("Gold Answer", ""),
            "replaced_sessions": replaced_sessions,  # 记录被替换的会话
            "recall_rate": 0.3  # 记录召回率
        }
        results.append(result)
        # 保存中间结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved intermediate results to {output_file}")
    print(f"\nProcessing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()