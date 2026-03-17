import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ベースモデルと学習済みアダプター
base_model_name = "Qwen/Qwen3-1.7B"
adapter_path = "./roadster-qwen3-lora-rtx5090"

print("🚗 ロードスターAIをロード中...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)

def ask_roadster(question):
    messages = [
        {"role": "system", "content": "あなたはNDロードスターの改造専門家です。"},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# テスト質問
test_questions = [
    "2016年式ND5RC（990S）にビルシュタイン車高調は適合しますか？",
    "NDロードスターのECUチューニング、おすすめのショップは？",
    "予算30万円で、一番走りが変わる改造は？"
]

for q in test_questions:
    print(f"\n❓ {q}")
    print(f"🤖 {ask_roadster(q)}")
    print("-" * 50)