"""
修正版：JSONエラー処理を強化した高速生成スクリプト
"""
import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

load_dotenv(encoding='utf-8-sig')

# ============================================
# 設定
# ============================================
BATCH_SIZE = 50  # 一旦50に戻して安定化
MAX_RETRIES = 3
RETRY_DELAY = 5

class RobustDataGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            timeout=120
        )
        self.generated_count = 0
        self.start_time = time.time()
        self.error_count = 0
        
    def extract_json_from_response(self, content):
        """レスポンスからJSONを抽出（エラー処理強化）"""
        try:
            # 直接パースを試みる
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.warning(f"JSONパース失敗: {e}")
            
            # JSON部分を正規表現で抽出
            json_pattern = r'(\[.*\]|\{.*\})'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    # 制御文字を除去
                    clean_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', match)
                    return json.loads(clean_json)
                except:
                    continue
            
            # それでもダメなら空リスト
            return []
    
    def generate_batch(self, category, size):
        """1バッチを生成（エラー処理強化）"""
        
        # より明確なプロンプト
        prompt = f"""あなたはNDロードスターの改造専門家です。
以下の指示に従って、必ず有効なJSON形式で出力してください。

【タスク】
{category}に関する質問と回答を{size}件生成してください。

【重要】
- 出力は必ずJSON配列形式にしてください
- 各要素は question, answer, category フィールドを含むこと
- 質問は具体的で実践的な内容に
- 回答は詳細で役立つ情報を含むこと
- 日本語で出力すること

【出力例】
[
  {{
    "category": "{category}",
    "question": "2016年式ND5RC（990S）にビルシュタイン車高調は適合しますか？",
    "answer": "適合します。ビルシュタイン製車高調（型番：XX-XXX）は2015年以降の全グレードに装着可能です。取付には...",
    "metadata": {{
      "grade": "990S",
      "year": "2016"
    }}
  }}
]

必ず上記のような有効なJSON配列のみを出力してください。説明文は不要です。"""

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "あなたは有効なJSONのみを出力するアシスタントです。説明は一切不要で、JSON配列のみを返してください。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                content = response.choices[0].message.content
                
                # JSON抽出
                data = self.extract_json_from_response(content)
                
                if data and isinstance(data, list):
                    self.generated_count += len(data)
                    self.error_count = 0  # 成功したらエラーカウントリセット
                    
                    # 速度計算
                    elapsed = time.time() - self.start_time
                    rate = self.generated_count / elapsed * 3600 if elapsed > 0 else 0
                    
                    logging.info(f"✓ {category}: {len(data)}件生成 (累計:{self.generated_count}件, 速度:{rate:.0f}件/時)")
                    
                    return data
                else:
                    logging.warning(f"空のデータを受信 (試行 {attempt + 1}/{MAX_RETRIES})")
                    
            except Exception as e:
                self.error_count += 1
                logging.error(f"エラー (試行 {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logging.info(f"{wait_time}秒後にリトライ...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"最大リトライ到達、スキップ")
                    
            time.sleep(2)  # レート制限対策
        
        return []  # すべて失敗したら空リスト

def main():
    print("""
    ╔══════════════════════════════════════╗
    ║  修正版 NDロードスター特化AI生成     ║
    ║  エラー処理強化版                     ║
    ╚══════════════════════════════════════╝
    """)
    
    # APIキーチェック
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ APIキーが設定されていません")
        print("以下のコマンドで設定してください：")
        print("set DEEPSEEK_API_KEY=sk-あなたのキー")
        return
    
    print(f"✅ APIキー: {api_key[:10]}... 確認済み")
    
    generator = RobustDataGenerator()
    
    # カテゴリと目標件数
    categories = [
        ("ecu_tuning", 1250),
        ("suspension", 1250),
        ("compatibility", 1000),
        ("troubleshooting", 750),
        ("general", 750)
    ]
    
    all_data = []
    
    try:
        for category, target in categories:
            logging.info(f"=== {category} 開始 (目標:{target}件) ===")
            
            generated = 0
            retry_count = 0
            
            while generated < target:
                # 残り件数に応じてバッチサイズ調整
                batch_size = min(BATCH_SIZE, target - generated)
                
                data = generator.generate_batch(category, batch_size)
                
                if data:
                    all_data.extend(data)
                    generated += len(data)
                    retry_count = 0
                    
                    # 100件ごとに保存
                    if len(all_data) % 100 == 0:
                        filename = f'nd_data_{len(all_data)}.jsonl'
                        with open(filename, 'w', encoding='utf-8') as f:
                            for item in all_data:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        logging.info(f"💾 保存: {filename} ({len(all_data)}件)")
                else:
                    retry_count += 1
                    if retry_count > 5:
                        logging.error(f"連続失敗、カテゴリ {category} をスキップ")
                        break
                
                # 短い待機
                time.sleep(1)
            
            logging.info(f"✓ {category} 完了: {generated}件")
        
        # 最終保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f'nd_roadster_complete_{timestamp}.jsonl'
        
        with open(final_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # CSV形式でも保存
        import csv
        csv_file = f'nd_roadster_complete_{timestamp}.csv'
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'question', 'answer', 'metadata'])
            for item in all_data:
                writer.writerow([
                    item.get('category', ''),
                    item.get('question', ''),
                    item.get('answer', ''),
                    json.dumps(item.get('metadata', {}), ensure_ascii=False)
                ])
        
        # 統計
        total_time = time.time() - generator.start_time
        hours = total_time / 3600
        avg_speed = len(all_data) / hours if hours > 0 else 0
        error_rate = generator.error_count / (generator.generated_count + generator.error_count) * 100
        
        logging.info("="*50)
        logging.info(f"🎉 完了: {len(all_data)}件")
        logging.info(f"⏱️ 総時間: {hours:.1f}時間")
        logging.info(f"⚡ 平均速度: {avg_speed:.0f}件/時間")
        logging.info(f"⚠️ エラー率: {error_rate:.1f}%")
        logging.info(f"💰 推定コスト: {(generator.generated_count * 2000 / 1000000 * 0.42 * 150):.0f}円")
        logging.info(f"📁 保存ファイル: {final_file}")
        logging.info(f"📁 CSVファイル: {csv_file}")
        
    except KeyboardInterrupt:
        logging.info("🛑 中断されました")
        # 中断時も保存
        if all_data:
            backup_file = 'nd_roadster_interrupted.jsonl'
            with open(backup_file, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logging.info(f"💾 中断データ保存: {backup_file} ({len(all_data)}件)")

if __name__ == "__main__":
    main()