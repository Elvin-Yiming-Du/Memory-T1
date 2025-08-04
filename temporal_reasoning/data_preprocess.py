import json
import argparse
from pathlib import Path

def build_cot_prompt(question, evidence, full_sessions):
    # 获取问题、时间戳、推理范围
    timestamp = evidence[0].get("utterance_id", "N/A")
    session_id = evidence[0].get("session_id", "N/A")
    question_line = f"{question} + timestamp={timestamp} + session={session_id}"

    # 构建 Retrieved History（格式：[timestamp utterance, ...]）
    retrieved_history = []
    for ev in evidence:
        ts = ev.get("utterance_id", "N/A")
        text = f"{ev.get('speaker', '')}: {ev.get('utterance', '')}"
        retrieved_history.append(f"{ts} {text}")
    retrieved_text = "[\n  " + ",\n  ".join(retrieved_history) + "\n]"

    # 构建 Selected Memory（所有出现过的 session）
    selected_sessions = sorted(set(ev["session_id"] for ev in evidence))
    selected_memory = []
    for sid in selected_sessions:
        if sid in full_sessions:
            selected_memory.append(f"<Session {sid}>\n{full_sessions[sid]}")
        else:
            selected_memory.append(f"<Session {sid}>")

    selected_memory_text = "\n\n".join(selected_memory)

    # 构建 response（单个答案）
    answer = evidence[0].get("answer", "")  # 可自定义读取
    response = f"The correct answer is: {answer}"

    # 合并为完整 prompt
    full_prompt = (
        f"### Question\n{question_line}\n\n"
        f"### Retrieved History\n{retrieved_text}\n\n"
        f"### Selected Memory\n{selected_memory_text}\n\n"
        f"### Response\n{response}"
    )
    return full_prompt


def convert_time_dial_to_verl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        time_dial_data = json.load(f)

    verl_data = []
    for item in time_dial_data:
        question_text = item["Question"]
        context_lines = []
        for ev in item.get("Evidence", []):
            context_lines.append(f'{ev["speaker"]}: {ev["utterance"]}')
        context = "\n".join(context_lines)

        full_prompt = (
            f"Read the following conversation context and answer the question.\n\n"
            f"**Question:** {question_text}\n\n"
            f"**Context:** {context}\n\n"
            f"Select the correct option (A, B, C, or D)."
        )

        verl_item = {
            "data_source": item.get("Dataset Name", "TIME-Dial"),
            "prompt": [
                {
                    "content": full_prompt,
                    "role": "user"
                }
            ],
            "ability": item.get("Task", "Explicit_Reasoning"),
            "reward_model": {
                "ground_truth": item["Gold Answer"],
                "style": "rule-lighteval/TIME-Dial_v1"
            },
            "extra_info": {
                "index": item["Question ID"]
            }
        }
        verl_data.append(verl_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verl_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully converted {len(verl_data)} examples to VERL format at {output_path}")

# CLI usage
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Convert TIME-Dial QA data to VERL input format.")
    # parser.add_argument("--input", type=str, required=True, help="Path to the input TIME-Dial JSON file.")
    # parser.add_argument("--output", type=str, required=True, help="Path to the output VERL JSON file.")
    # args = parser.parse_args()

    input_file = "/home/ubuntu/temporal_reasoning/time_dial_context_list_with_evidence_200.json"
    output_file = "/home/ubuntu/temporal_reasoning/time_dial_context_list_with_evidence_200_verl.json"

    convert_time_dial_to_verl(input_file, output_file)
