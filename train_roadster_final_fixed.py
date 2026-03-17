"""
RTX 5090用 ロードスターチューニングスクリプト（引数修正版）
"""
import os
# Xet無効化
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
import time

# ============================================
# 設定
# ============================================
MODEL_NAME = "Qwen/Qwen3-1.7B"  # または "./qwen-model"（ローカル保存した場合）
TRAIN_FILE = "roadster_train.jsonl"
VAL_FILE = "roadster_val.jsonl"
OUTPUT_DIR = "./roadster-qwen3-lora-rtx5090"

def main():
    print("🚗 NDロードスター特化AI 学習開始 (RTX 5090)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # トークナイザー読み込み
    print("📥 トークナイザー読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル読み込み
    print("📥 モデル読み込み中...（2-3分かかります）")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
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
    
    # 【修正ポイント】evaluation_strategy → eval_strategy
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",          # ← ここを修正！
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
    
    print("\n🚀 学習開始！")
    print("予想時間: 約10-15分")
    trainer.train()
    
    print("💾 モデル保存中...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ モデル保存完了: {OUTPUT_DIR}")
    
    # 簡単なテスト
    print("\n🧪 テスト生成実行中...")
    test_question = "2016年式のND5RC（990S）にビルシュタイン車高調は適合しますか？"
    messages = [
        {"role": "system", "content": "あなたはNDロードスターの改造専門家です。"},
        {"role": "user", "content": test_question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n❓ {test_question}")
    print(f"🤖 {response}")

if __name__ == "__main__":
    main()