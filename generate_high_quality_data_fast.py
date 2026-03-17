"""
高速・高品質データ生成スクリプト（DeepSeek V3.2最適化版）
特徴：
- 並列リクエストで高速化
- 具体性を強制するプロンプト
- 実在するショップ・製品名の活用
- 進捗の可視化と自動保存
"""

import os
import json
import time
import asyncio
import aiohttp
from openai import AsyncOpenAI
from datetime import datetime
from tqdm.asyncio import tqdm
import pandas as pd
from typing import List, Dict, Any
import random

# ============================================
# 設定
# ============================================
CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "max_concurrent": 5,  # 同時リクエスト数（高速化の核心）
    "temperature": 0.8,
    "max_tokens": 4096,
    "timeout": 120
}

# 実在するショップ・メーカーリスト（具体性向上のため）
SHOPS = [
    "HKS関東サービス", "トラスト/GReddy", "ブリッツ", "BLITZ", 
    "AutoExe", "マツダスピード", "Craft-web", "タイヤセレクト大宮",
    "BSCフジノ", "ロードスターチューニングファクトリー", "NDカスタム"
]

PRODUCTS = {
    "suspension": [
        "ビルシュタインB14", "ビルシュタインB16", "オーリンズDFV",
        "HKSハイパーマックスS", "KWバリアント3", "クスコストリートZERO"
    ],
    "ecu": [
        "HKSフラッシュエディター", "トラストF-CON V Pro", "コムテックiQ",
        "エキマニ", "スポーツキャタライザー"
    ],
    "exhaust": [
        "HKSリーガマックスプレミアム", "HKSハイパワーマフラー",
        "AutoExeマフラー", "マツダスピードマフラー"
    ]
}

# カテゴリと目標件数（1,000件追加用）
TARGET_CATEGORIES = {
    "suspension": 300,      # 足回り
    "ecu_tuning": 300,      # ECUチューニング
    "compatibility": 200,   # 適合情報
    "troubleshooting": 100, # トラブルシュート
    "abstract_request": 100 # 抽象的要望（「走りを良くしたい」系）
}

# ============================================
# プロンプトテンプレート
# ============================================
SYSTEM_PROMPT = """あなたはマツダ・ロードスターND型（2015年〜現行）の改造・チューニング専門家です。
以下の厳守事項を必ず守ってください：

1. 年式は正確に（NDは2015年〜現行、年次改良あり）
2. 実在するショップ・メーカー名のみを使用
3. 具体的な価格・工賃を含める
4. 安全に関する注意点を必ず記載
5. 「例えば」「具体的には」を使って説明

出力は必ず有効なJSON配列形式とし、説明文は一切含めないこと。"""

def get_prompt(category: str, count: int = 10) -> str:
    """カテゴリ別プロンプト（具体性重視）"""
    
    base_prompt = f"""NDロードスターの{category}に関する具体的な質問と回答を{count}件、JSON配列で生成してください。

【必須要素】
- 年式（例：2016年式、2019年式、2024年改良型）
- グレード（990S、RS、NR-A、RF、12R、MSRスペシャル）
- 実在するショップ名（{', '.join(random.sample(SHOPS, 3))}など）
- 実在する製品名（{', '.join(random.sample(PRODUCTS.get(category.split('_')[0], PRODUCTS['suspension']), 2))}など）
- 具体的な価格（例：18万円、工賃3-5万円）
- 安全に関する注意点

【出力形式】
[
  {{
    "category": "{category}",
    "question": "ユーザーからの具体的な質問文",
    "answer": "専門家による詳細な回答（具体的情報を含む）",
    "metadata": {{
      "year": "対象年式",
      "grade": "対象グレード",
      "shop": "参考になるショップ",
      "price_range": "価格帯"
    }}
  }}
]"""

    # カテゴリ別の追加指示
    if category == "abstract_request":
        base_prompt += """

【抽象的要望の例】
- 「峠を気持ちよく走りたいんだけど、何から手をつければいい？」
- 「サーキットデビューしたいけど、今の車で大丈夫？」
- 「予算30万円で、一番走りが変わる改造は？」

回答では具体的な改造手順と予算配分を示すこと。"""
    
    elif category == "troubleshooting":
        base_prompt += """

【トラブル例】
- ECU書き換え後のアイドリング不安定
- 足回り交換後の異音
- 警告灯点灯

原因推定と具体的な対策、修理費用の目安を含めること。"""
    
    return base_prompt

# ============================================
# 非同期データ生成クラス
# ============================================
class AsyncDataGenerator:
    """非同期処理による高速データ生成"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=CONFIG["api_key"],
            base_url=CONFIG["base_url"],
            timeout=CONFIG["timeout"]
        )
        self.generated = []
        self.total_tokens = 0
        self.start_time = time.time()
        self.semaphore = asyncio.Semaphore(CONFIG["max_concurrent"])
    
    async def generate_one_batch(self, category: str, batch_size: int) -> List[Dict]:
        """1バッチを非同期で生成"""
        
        async with self.semaphore:  # 同時実行数を制限
            try:
                response = await self.client.chat.completions.create(
                    model=CONFIG["model"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": get_prompt(category, batch_size)}
                    ],
                    temperature=CONFIG["temperature"],
                    max_tokens=CONFIG["max_tokens"],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                self.total_tokens += response.usage.total_tokens
                
                # JSON抽出（エラー処理強化）
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
                except:
                    # JSONが壊れている場合の救済処理
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        return json.loads(match.group())
                
                return []
                
            except Exception as e:
                print(f"エラー: {e}")
                return []
    
    async def generate_category(self, category: str, target: int, pbar):
        """カテゴリ単位での生成"""
        
        tasks = []
        batch_size = 20  # 1バッチ20件（高速化と安定性のバランス）
        
        for _ in range(0, target, batch_size):
            current_batch = min(batch_size, target - len(self.generated))
            if current_batch <= 0:
                break
            
            task = self.generate_one_batch(category, current_batch)
            tasks.append(task)
        
        # バッチを並列実行
        results = await asyncio.gather(*tasks)
        
        for batch_data in results:
            if batch_data:
                self.generated.extend(batch_data)
                pbar.update(len(batch_data))
                
                # 100件ごとに保存
                if len(self.generated) % 100 == 0:
                    self.save_interim()
    
    def save_interim(self):
        """途中経過保存"""
        filename = f"interim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.generated:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n💾 保存: {filename} ({len(self.generated)}件)")

# ============================================
# メイン実行
# ============================================
async def main():
    print("""
    ╔══════════════════════════════════════════════╗
    ║  高速・高品質 ロードスターデータ生成 v2.0   ║
    ║         DeepSeek V3.2 並列処理版             ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # APIキーチェック
    if not CONFIG["api_key"]:
        print("❌ APIキーが設定されていません")
        print("set DEEPSEEK_API_KEY=sk-xxx を実行してください")
        return
    
    print(f"🚀 同時実行数: {CONFIG['max_concurrent']}")
    print(f"🎯 目標件数: {sum(TARGET_CATEGORIES.values())}件")
    print("内訳:")
    for cat, num in TARGET_CATEGORIES.items():
        print(f"  - {cat}: {num}件")
    
    # ユーザー確認
    response = input("\n生成を開始しますか？ (y/n): ")
    if response.lower() != 'y':
        return
    
    generator = AsyncDataGenerator()
    
    # 全体の進捗バー
    total_target = sum(TARGET_CATEGORIES.values())
    with tqdm(total=total_target, desc="総合進捗") as pbar:
        for category, target in TARGET_CATEGORIES.items():
            print(f"\n📝 {category} 生成開始...")
            await generator.generate_category(category, target, pbar)
    
    # 最終保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"roadster_high_quality_{timestamp}.jsonl"
    
    with open(final_file, 'w', encoding='utf-8') as f:
        for item in generator.generated:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 統計表示
    elapsed = time.time() - generator.start_time
    print(f"\n" + "="*50)
    print(f"🎉 生成完了: {len(generator.generated)}件")
    print(f"⏱️ 処理時間: {elapsed/60:.1f}分")
    print(f"⚡ 生成速度: {len(generator.generated)/elapsed*3600:.0f}件/時間")
    print(f"💰 推定コスト: {generator.total_tokens/1_000_000 * 0.42 * 150:.0f}円")
    print(f"📁 保存ファイル: {final_file}")
    
    # CSVでも保存
    df = pd.DataFrame(generator.generated)
    csv_file = final_file.replace('.jsonl', '.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"📁 CSV保存: {csv_file}")

if __name__ == "__main__":
    asyncio.run(main())