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

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    计算输入字符串的 token 数量

    Args:
        text: 要计算的字符串
        model: 使用的模型名称（决定分词规则）

    Returns:
        token 数量
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def load_time_contexts():
    """加载带时间标注的对话文件"""
    context_file = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_labeled_contexts.json"
    with open(context_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_session_context(session_id, session: Dict) -> str:
    """将会话格式化为文本"""
    context = f"{session.get('session_date')}: <{str(session_id)}>, "
    utterances = []
    for utterance in session.get('content', []):
        utterance_text = f"{utterance.get('speaker')} {utterance.get('utterance_date')} : {utterance.get('utterance')}"
        utterances.append(utterance_text)
    context += "\n".join(utterances)
    return context

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

def build_prompt(question, dialogue_sessions, now):
    response_format =  f"""You are a memory-aware assistant tasked with answering temporal questions based on multi-turn dialogue history.
Output JSON MUST include:
1. "selected_memory": a list of memory identifiers (e.g., "<session_1>") or quoted utterances that are relevant to answering the question.
2. "answer": must strictly follow one of the formats below:
    - Single choice: "A", "B", etc.
    - Multiple choice: "A B C", "B D", etc.
    - Time: "10:45:41 pm, January 15, 2024"
    - Sequence: "(1)(4)(3)(2)(5)(6)(8)(7)"

Important: When analyzing the dialogue history:
- Pay close attention to the temporal relationships between events
- Consider the chronological order of events
- Identify explicit and implicit time references
- Ensure the answer is consistent with selected memory and the temporal context

**User Question:**
{str(now)}: {question}

***session memory:** 
{dialogue_sessions}

**Expected Response:**
```json
{{
  "selected_memory": ["<relevant_session_1>", "<relevant_session_2>"],
  "answer": "<your generated response in the exact required format, like "A","Friday","B C", "10:45:41 pm, January 15, 2024" OR "(1)(4)(3)(2)">"
}}
```
"""
    return response_format

def infer_with_local_model(prompt, tokenizer, model, temperature=0.2, top_p=0.95, max_new_tokens=512, stop=None):

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

    # === 4. 加载 base Qwen 模型 ===
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True  # Qwen 必须设置为 True
    )

    # === 5. 加载 LoRA adapter ===
    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return tokenizer, model

def main():
    # ====== 路径配置（请根据实际情况修改） ======
    time_contexts = load_time_contexts()
    question_file = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_context_list_with_evidence_200_v2.json"  # 问题列表，每个元素至少有Question字段
    output_file = "./bm25_local_infer_results_new2.json"
    max_prompt_length = 15500
    base_model_path = "/home/ubuntu/cloud_disk/models/Qwen2.5-3B-Instruct"  # 基础模型路径
    lora_adapter_path = "/home/ubuntu/cloud_disk/verl_main/checkpoints/time-qwen-2.5-3b-instruct/time-qwen-2.5-3b-instruct-sp2/global_step_477/actor"

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
        top_ids = bm25_retriever(query, evidence_sessions, top_k=10)
        # Recall@5: gold_sessions为list，统计每个gold session是否在top5_ids中
        retrieved_sessions = []
        load_sessions_length = 0

        for session_id in top_ids:
            formatted_session_content = format_session_context(session_id, evidence_sessions[session_id])
            load_sessions_length += count_tokens(formatted_session_content)
            if load_sessions_length < max_prompt_length:
                retrieved_sessions.append(formatted_session_content)
            else:
                break

        if gold_sessions:
            for evident_session in gold_sessions:
                if evident_session["session_id"] in top_ids:
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
            "bm25_top5_session_ids": top_ids,
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