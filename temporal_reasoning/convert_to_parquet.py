import pandas as pd
import json
import random

# ===== 配置区 =====
json_path = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_context_with_evidence_new_updated_0723_verl.json"
parquet_path = "/home/ubuntu/cloud_disk/temporal_reasoning/time_dial_context_with_evidence_new_updated_0723_verl.parquet"

# 是否分为train/val
split_train_val = True  # 设置为True则分割
train_ratio = 0.9       # 训练集比例
train_parquet_path = parquet_path.replace('.parquet', '_train.parquet')
val_parquet_path = parquet_path.replace('.parquet', '_val.parquet')
# ===================

# 读取 JSON 文件
with open(json_path, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except UnicodeDecodeError:
        with open(json_path, "r", encoding="gbk") as f2:
            data = json.load(f2)

# 分割数据（可选）
if split_train_val:
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"数据总数: {len(data)}, 训练集: {len(train_data)}, 验证集: {len(val_data)}")
    data_splits = [(train_data, train_parquet_path), (val_data, val_parquet_path)]
else:
    data_splits = [(data, parquet_path)]

expected_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

def parse_extra_info(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except:
            return val
    return val

for split_data, out_path in data_splits:
    df = pd.DataFrame(split_data)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    df = df[expected_columns]
    # 保证 extra_info 列全部为字符串
    df["extra_info"] = df["extra_info"].apply(lambda x: json.dumps(parse_extra_info(x), ensure_ascii=False))
    df.to_parquet(out_path, index=False)
    print(f"转换完成 ✅ 保存到: {out_path}")