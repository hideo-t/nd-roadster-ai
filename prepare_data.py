"""
ロードスターデータをQwen3学習用に変換
"""
import json
import random
from datetime import datetime

# システムプロンプト（AIの役割定義）
SYSTEM_PROMPT = """あなたはマツダ・ロードスターND型（2015年〜現行）の改造・チューニング専門家です。
年式・グレードを考慮した正確な情報を提供し、安全に関する注意点も必ず伝えてください。"""

# 元データの読み込み
input_file = "nd_roadster_complete_20260314_080056.jsonl"  # あなたのファイル名
train_file = "roadster_train.jsonl"
val_file = "roadster_val.jsonl"

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# ランダムにシャッフル
random.shuffle(data)

# 訓練:検証 = 9:1 に分割
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"📊 データ件数: 合計 {len(data)}件")
print(f"   - 訓練: {len(train_data)}件")
print(f"   - 検証: {len(val_data)}件")

# Qwen3用フォーマットに変換
def convert_to_qwen_format(item):
    """Qwen3のチャットテンプレート形式に変換"""
    question = item.get('question', '')
    answer = item.get('answer', '')
    
    # Qwen3の形式: <|im_start|>system...<|im_end|> <|im_start|>user...<|im_end|> <|im_start|>assistant...
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    # JSONL形式で保存するため、そのままJSONとして保持
    return {"messages": messages}

# 訓練データ保存
with open(train_file, 'w', encoding='utf-8') as f:
    for item in train_data:
        converted = convert_to_qwen_format(item)
        f.write(json.dumps(converted, ensure_ascii=False) + '\n')

# 検証データ保存
with open(val_file, 'w', encoding='utf-8') as f:
    for item in val_data:
        converted = convert_to_qwen_format(item)
        f.write(json.dumps(converted, ensure_ascii=False) + '\n')

print("✅ 変換完了: roadster_train.jsonl / roadster_val.jsonl")