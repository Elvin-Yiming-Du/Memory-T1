import json
import os
import argparse
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
import tiktoken
import time
import torch
import torch_npu  # 导入NPU支持库
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

def format_session_context(session_id, session: Dict) -> str:
    """将会话格式化为文本"""
    context = f"{session.get('session_date')}: <{str(session_id)}>, "
    utterances = []
    for utterance in session.get('content', []):
        utterance_text = f"{utterance.get('speaker')} : {utterance.get('utterance_date')} : {utterance.get('utterance')}"
        utterances.append(utterance_text)
    context += "\n".join(utterances)
    return context

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

def infer_with_local_model(prompt, tokenizer, model, max_prompt_length=15500, temperature=0.1, top_p=0.95, max_new_tokens=512, stop=None):

    prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # positional argument
            add_generation_prompt=True,
            tokenize=False
        )
    
    # 先估算token数量
    estimated_tokens = len(tokenizer.encode(prompt_text))
    print(f"Estimated tokens: {estimated_tokens}")
    if estimated_tokens < 1000:
        print(prompt_text)
    
    inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=max_prompt_length)
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
    return answer
    
# def parse_model_output(output):
#     match = re.search(r'\{[\s\S]*\}', output)
#     if match:
#         try:
#             return json.loads(match.group(0))
#         except Exception:
#             return {}
#     return {}

def parse_model_output(output):
    # match = re.search(r'\{[\s\S]*\}', output)
    res = output.replace("Answer: ","")
    return res

def load_model_and_tokenizer(base_model_path, lora_model_path="", target_device = "npu:0"):
    """加载LoRA模型和分词器 - 保守版本"""
    from peft import PeftModel
    
    # === 加载 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # === 加载 base Qwen 模型到指定设备 ===
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=None,  # 禁用自动设备映射
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        do_sample=False,
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
    
    
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="时间推理模型参数配置")
    
    
    parser.add_argument("--question_file", type=str, required=False, default="/home/ma-user/work/ydu/data/time_range/time_dial_context_list_with_evidence_200_v2.json",
                       help="问题列表文件路径，每个元素至少有Question字段")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出结果文件路径")
    parser.add_argument("--max_prompt_length", type=int, default=7500,
                       help="最大提示长度，默认为7500")
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="基础模型路径")
    parser.add_argument("--target_device", type=str, required=True, default="npu:0",
                       help="基础模型路径")
    
    # 解析参数
    args = parser.parse_args()
    
    question_file = args.question_file
    output_file = args.output_file
    max_prompt_length = args.max_prompt_length
    base_model_path = args.base_model_path
    npu_target_device = args.target_device
    # ====== 路径配置（请根据实际情况修改） ======
    time_contexts = load_time_contexts()
    # output_file = "/home/ma-user/work/ydu/temporal_reasoning/results/Qwen2.5_7B_Instruct-results_8k_0905.json"
    # max_prompt_length= 7500
    # base_model_path = "/home/ma-user/work/ydu/models/Qwen/Qwen2.5_7B_Instruct"  # 基础模型路径

    tokenizer, model = load_model_and_tokenizer(base_model_path, target_device=npu_target_device)

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
        session_context = question.get('Context', '')
        evidence_sessions = time_contexts[session_context]
        full_sessions = []
        load_sessions_length = 0
        for sesssion_id, session in reversed(list(evidence_sessions.items())):
            formatted_session_content = format_session_context(sesssion_id, session)
            load_sessions_length += count_tokens(formatted_session_content)
            if load_sessions_length < max_prompt_length:
                full_sessions.append(formatted_session_content)
            else:
                break
        CURRENT_TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 2. 本地模型推理
        prompt = build_prompt(query, "\n".join(full_sessions), CURRENT_TIME)
        answer = infer_with_local_model(prompt, tokenizer, model, max_prompt_length)
        # print(answer)
        parsed = parse_model_output(answer)
        print("parsed_result:",parsed)
        print("gold_answer:",question.get("Gold Answer", ""))
        result = {
            "question_id": question.get("Question ID"),
            "question": query,
            "level": question.get("Level"),
            "task": question.get("Task"),
            "prompt": prompt,
            "model_output": answer,
            "parsed_output": parsed,
            "gold_answer": question.get("Gold Answer", "")
        }
        results.append(result)
        # 保存中间结果
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