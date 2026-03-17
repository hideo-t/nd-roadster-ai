import subprocess
import sys

def check_gpu_memory_and_warn(threshold_gb=24):
    """
    GPUメモリ使用量をチェックし、閾値を超えていたらユーザーに確認する。
    nvidia-smiコマンドを使用する（Windowsの場合）。
    """
    print("🔍 GPUメモリ使用状況をチェック中...")
    try:
        # nvidia-smi を実行してGPU使用プロセスを取得
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:  # 結果が空でなければ
            print("以下のプロセスがGPUメモリを使用しています:")
            for line in lines:
                print(f"  {line.strip()}")

            # メモリ使用量の合計を計算（簡易版）
            total_used = 0
            for line in lines:
                try:
                    # カンマで分割し、最後の要素がメモリ使用量(MiB)
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        mem_str = parts[2].strip()
                        if mem_str.isdigit():
                            total_used += int(mem_str)
                except Exception:
                    pass

            print(f"合計使用メモリ: {total_used} MiB / {threshold_gb*1024} MiB (閾値)")

            if total_used > threshold_gb * 1024:
                print("⚠️ 警告: 他のアプリケーションが大量のGPUメモリを使用しています。")
                response = input("このまま学習を続けるとメモリ不足で失敗する可能性があります。続行しますか？ (y/N): ")
                if response.lower() != 'y':
                    print("ユーザーにより中断されました。他のアプリを終了してから再実行してください。")
                    sys.exit()
            else:
                print("✅ GPUメモリに余裕があります。学習を開始します。")
        else:
            print("✅ GPUを使用している他のプロセスは見つかりませんでした。")

    except FileNotFoundError:
        print("⚠️ nvidia-smi が見つかりません。GPUの状態を確認できません。")
    except Exception as e:
        print(f"⚠️ GPUチェック中にエラーが発生しました: {e}")

# --- この関数を main() の最初の方で呼び出す ---
def main():
    # 追加: GPUメモリチェック（閾値は24GBに設定）
    check_gpu_memory_and_warn(threshold_gb=24)

    # 以降、通常のモデル読み込みと学習処理...
    print("🚗 NDロードスター特化AI 学習開始 (RTX 5090専用)")
    # ... (以下、既存のコード)


"""
RTX 5090用：CPUオフロード完全防止版
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 【重要】GPU強制設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # メモリ効率化

import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# GPUが使えるか確認
print(f"🚀 GPU利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    # デバイスを明示的にcuda:0に設定
    device = torch.device("cuda:0")
else:
    raise RuntimeError("GPUが見つかりません！RTX 5090を認識できていません")

# ============================================
# 設定
# ============================================
MODEL_NAME = "Qwen/Qwen3-1.7B"  # またはローカルパス
TRAIN_FILE = "roadster_train.jsonl"
VAL_FILE = "roadster_val.jsonl"
OUTPUT_DIR = "./roadster-qwen3-lora-rtx5090"

def main():
    print("🚗 NDロードスター特化AI 学習開始 (RTX 5090専用)")
    
    # トークナイザー読み込み
    print("📥 トークナイザー読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み（GPU強制）
    print("📥 モデル読み込み中...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",           # ← GPUに強制配置
        trust_remote_code=True
    )
    
    # モデルがGPUにあることを確認
    print(f"✅ モデル配置: {next(model.parameters()).device}")
    
    # LoRA設定
    print("🔧 LoRA設定中...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    # データセット読み込み
    print("📂 データセット読み込み中...")
    dataset = load_dataset('json', data_files={
        'train': TRAIN_FILE,
        'validation': VAL_FILE
    })
    
    # 前処理関数
    def preprocess_function(examples):
        texts = []
        for messages in examples["messages"]:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(formatted)
        
        model_inputs = tokenizer(
            texts,
            max_length=1024,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    print("🔄 データ前処理中...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    print(f"✅ 訓練データ: {len(tokenized_dataset['train'])}件")
    print(f"✅ 検証データ: {len(tokenized_dataset['validation'])}件")
    
    # 【重要】CPUオフロードを防ぐTrainingArguments設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        learning_rate=3e-4,
        fp16=True,
        bf16=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        # GPU最適化設定
        optim="adamw_torch",           # PyTorch標準の最適化
        gradient_checkpointing=False,   # RTX 5090なら不要
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\n🚀 学習開始！（すべてGPUで実行中）")
    print(f"  バッチサイズ: 8 (実効: 16)")
    print("  Ctrl+Cで中断すると途中保存されます")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 中断されました。途中まで保存します...")
    
    print("💾 モデル保存中...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ モデル保存完了: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()