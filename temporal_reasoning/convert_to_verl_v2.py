import json
import re
import random
from datetime import datetime
from this import d
import tiktoken
from tqdm import tqdm

def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def format_sessions(sessions):
    lines = []
    for s in sessions:
        utterances = " ".join(u['speaker'] + ": " + u['utterance_time'] + ": " + u['text'] + "\n" for u in s['utterances'])
        lines.append(f"{s['session_time']}: <{s['session_id']}> \n {utterances}")
    return "\n".join(lines)

def build_dialogue_history_with_token_budget(gold_sessions, non_gold_sessions, question_text, now, max_tokens=3500):
    # 1. 先拼接gold session
    selected_sessions = gold_sessions.copy()
    # 2. 随机打乱non-gold
    random.shuffle(non_gold_sessions)
    # 3. 依次尝试加入non-gold session，直到超预算
    for session in non_gold_sessions:
        candidate_sessions = selected_sessions + [session]
        dialogue_history = format_sessions(candidate_sessions)
        if count_tokens(dialogue_history) > max_tokens:
            break
        selected_sessions.append(session)
    # 4. 最终随机打乱所有session
    random.shuffle(selected_sessions)
 
    dialogue_history = format_sessions(selected_sessions)

    return dialogue_history

now = datetime.now()
with open("/home/ubuntu/cloud_disk/temporal_reasoning/question_time_range_annotated_0717.json", "r", encoding="utf-8") as f:
    QUESTION_TIME_RANGE_MAP = json.load(f)

def extract_time_range_from_question(question_id):
    # 直接查表
    item = QUESTION_TIME_RANGE_MAP.get(str(question_id))
    if item and "question_time_range" in item:
        return item["question_time_range"]
    else:
        # 没找到时返回默认
        return {"start": "2021-01-01", "end": "2025-12-31"}


def build_cot_prompt(question, dialogue_history, now=None):
    response_format = f"""You are a memory-aware assistant tasked with answering temporal questions based on multi-turn dialogue history.
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
{dialogue_history}

**Expected Response:**
```json
{{
  "selected_memory": ["<relevant_session_1>", "<relevant_session_2>"],
  "answer": "<your generated response in the exact required format, like "A","Friday","B C", "10:45:41 pm, January 15, 2024" OR "(1)(4)(3)(2)">"
}}
```
"""
    return response_format

# def build_cot_prompt(question, dialogue_history, now=None):
#     response_format = f"""You are a memory-aware assistant tasked with answering temporal questions based on multi-turn dialogue history.
# Output JSON MUST include:
# 1. "selected_memory": a list of memory identifiers (e.g., "session_1") or quoted utterances that are relevant to answering the question.
# 2. "answer": must strictly follow one of the formats below:
#     - Single choice: "A", "B", etc.
#     - Multiple choice: "A B C", "B D", etc.
#     - Time: "10:45:41 pm, January 15, 2024"
#     - Sequence: "(1)(4)(3)(2)"

# Important: When analyzing the dialogue history:
# - Pay close attention to the temporal relationships between events
# - Consider the chronological order of events
# - Identify explicit and implicit time references
# - Ensure the answer is consistent with selected memory and the temporal context

# ---

# ### Example

# **User Question:**  
# 03:45:22 PM, July 18, 2025: When did India go to MoMA? A. Friday B. Saturday C. Sunday D. Monday

# **session memory:**  
# 03:45:22 PM, July 14, 2025: session_1: India: I went to the museum last Friday.  
#                                        Debra: Which one?  
# 05:12:51 PM, July 16, 2025: session_2: India: The MoMA.  
# 06:32:42 PM, July 17, 2025: session_3: Debra: That sounds fun.  
# 07:34:13 PM, July 18, 2025: session_4: India: I stayed there until 6 pm.

# **Expected Response:**
# ```json
# {{
#   "selected_memory": ["<session_1>", "<session_2>"],
#   "answer": "C"
# }}
# ```

# Now Your Turn

# **User Question:**
# {str(now)}: {question}

# ***session memory:** 
# {dialogue_history}

# **Expected Response:**
# ```json
# {{
#   "selected_memory": ["<relevant_session_or_utterance1>", "<relevant_session_or_utterance2>"],
#   "answer": "<your generated response in the exact required format>"
# }}
# ```
# """
#     return response_format

def extract_time_range_from_evnets(evnets):
    start_time = None
    end_time = None
    for evnet in evnets:
        if "event_time" in evnet:
            if start_time is None:
                start_time = evnet["event_time"][0]
            if end_time is None:
                end_time = evnet["event_time"][1]
            if evnet["event_time"][0]!="unknown" and evnet["event_time"][0] < start_time:
                start_time = evnet["event_time"][0]
            if evnet["event_time"][1]!="unknown" and evnet["event_time"][1] > end_time:
                end_time = evnet["event_time"][1]
    if start_time is None or end_time is None:
        return {"start": "unknown", "end": "unknown"}
    return {"start": start_time, "end": end_time}


def determine_session_relevance(session_id, evidence_sessions):
    return "yes" if str(session_id) in evidence_sessions else "no"

def determine_utterance_relevance(utterance, evidence_utterances):
    text = utterance.get("utterance", "")
    for ev in evidence_utterances:
        if ev.get("utterance", "") == text:
            return "yes"
    return "no"

def convert_to_verl_v2_format(input_path, output_path):
    context_file = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_labeled_contexts.json"
    with open(context_file, "r", encoding="utf-8") as f:
        labeled_contexts = json.load(f)
    with open(input_path, "r", encoding="utf-8") as f:
        time_dial_data = json.load(f)

    verl_data = []
    counter = 0
    for item in tqdm(time_dial_data):
        counter += 1
        if counter < 211:
            continue
        question_text = item["Question"]
        question_id = item.get("Question ID", "")
        gold_answer = item.get("Gold Answer", "")
        evidence_list = item.get("Evidence", [])
        evidence_sessions = set()
        evidence_utterances = []
        for evidence in evidence_list:
            session_id = evidence.get("session_id")
            if session_id:
                evidence_sessions.add(str(session_id))
            evidence_utterances.append(evidence)
        question_time_range = extract_time_range_from_question(question_id)
        context_key = item.get("Context", "")

        # 获取所有的session
        full_sessions = labeled_contexts.get(context_key, {})
        all_sessions = []
        for session_id, session_content in full_sessions.items():
            is_session_relevant = "yes" if str(session_id) in evidence_sessions else "no"
            utterances = []
            for utterance_item in session_content.get("content", []):
                is_utterance_relevant = "yes" if any(ev.get("utterance", "") == utterance_item.get("utterance", "") for ev in evidence_utterances) else "no"
                utterance_data = {
                    "speaker": utterance_item.get("speaker", "").split(" ")[0],
                    "text": utterance_item.get("utterance", ""),
                    "utterance_time": utterance_item.get("utterance_date", ""),
                    "event_time": extract_time_range_from_evnets(utterance_item.get("events", [])),
                    "is_relevant": is_utterance_relevant
                }
                utterances.append(utterance_data)
            session_data = {
                "session_id": f"session_{str(session_id)}",
                "session_time": session_content.get("session_date", ""),
                "is_relevant": is_session_relevant,
                "utterances": utterances
            }
            all_sessions.append(session_data)
        selected_sessions = sorted([f"session_{str(session_id)}" for session_id in evidence_sessions])

        # 1. gold session
        gold_session_ids = set(selected_sessions)
        gold_sessions = [s for s in all_sessions if s["session_id"] in gold_session_ids]
        non_gold_sessions = [s for s in all_sessions if s["session_id"] not in gold_session_ids]

        # 使用统一打乱+token预算拼接逻辑
        dialogue_history = build_dialogue_history_with_token_budget(
            gold_sessions, non_gold_sessions, question_text, now, max_tokens=15000
        )
        prompt = build_cot_prompt(question_text, dialogue_history, now)
        verl_item = {
            "data_source": item.get("Dataset Name", "TIME-Dial"),
            "prompt": [
                {
                    "content": prompt,
                    "role": "user"
                }
            ],
            "ability": item.get("Task", item.get("Task", "")),
            "reward_model": {
                 "ground_truth": {
                    "answer": gold_answer,
                    "selected_sessions": selected_sessions
                },
                "style": "rule-lighteval/TIME-Dial_v1"
            },
            "extra_info": {
                "index": item.get("Question ID", ""),
                "gold_selected_sessions": selected_sessions,
                "question_time_range": question_time_range,
                "all_sessions": all_sessions
            }
        }
        verl_data.append(verl_item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verl_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Successfully converted {len(verl_data)} examples to VERL v2 format at {output_path}")

if __name__ == "__main__":
    input_file = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_context_with_evidence_new_updated_0717.json"
    output_file = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_context_with_evidence_new_updated_0723_verl.json"
    convert_to_verl_v2_format(input_file, output_file)