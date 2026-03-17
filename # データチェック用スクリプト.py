# データチェック用スクリプト
import json

with open('roadster_train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 問題のあるデータを検出
issues = []
for i, item in enumerate(data):
    messages = item['messages']
    question = messages[1]['content']
    answer = messages[2]['content']
    
    # 年式の誤りチェック
    if '2017' in answer and 'ND' in answer:
        issues.append(f"行{i}: 年式誤りの可能性 - {answer[:100]}")
    
    # 抽象的回答チェック
    if len(answer) < 100:
        issues.append(f"行{i}: 回答が短すぎる - {answer}")

print(f"問題候補: {len(issues)}件")
for issue in issues[:10]:
    print(issue)