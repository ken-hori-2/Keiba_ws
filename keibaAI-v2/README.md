# 🐎 競馬AI予測システム v2 (keibaAI-v2)

## 📋 概要

keibaAI-v2は、netkeiba.comからのリアルタイムデータを活用した高精度競馬予測AIシステムです。LightGBMベースの機械学習モデルを使用して、**収益性193.1%を達成**した実戦的な投資判断支援システムです。

### 🎯 主要機能

- **📡 リアルタイムデータ取得**: netkeiba.comからの即座のスクレイピング
- **🤖 機械学習予測**: LightGBMベースの高精度予測モデル
- **📊 投資シミュレーション**: 実戦的な投資判断とリスク管理
- **📈 可視化・分析**: 詳細な分析結果とパフォーマンス評価
- **💡 自動レース選定**: 収益性の高いレースの自動識別

### 🏆 実績

- **収益率**: 193.1%達成
- **的中精度**: 高精度な上位馬予測
- **リアルタイム対応**: 当日レースの即座の分析

---

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境の作成と有効化
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 必要なライブラリのインストール
pip install requests beautifulsoup4 lxml lightgbm pandas numpy matplotlib seaborn scikit-learn
```

### 2. 基本的な使用方法

```python
# 今日のレース一覧を取得
today_races = scraper.get_race_calendar()

# 特定のレースを予測
results = predict_specific_race('202406050101')

# 結果を確認
print(results[['馬番', 'horse_name', 'prediction_proba', 'recommendation']])
```

### 3. 実行手順

1. `main.ipynb` または `horse_racing_ai_prediction.ipynb` を開く
2. 全セルを順次実行してシステムを初期化
3. `predict_specific_race('レースID')` で予測を実行

---

## 📁 ファイル構成

```
keibaAI-v2/
├── README.md                                    # プロジェクト概要（本ファイル）
├── main.ipynb                                   # メインの実行ノートブック
├── horse_racing_ai_prediction.ipynb            # 予測モデル開発用ノートブック
├── horse_racing_ai_prediction_real_data.ipynb  # 実データ検証用ノートブック
├── horse_racing_ai_user_guide.md               # 詳細な使用方法ガイド
└── .venv/                                       # Python仮想環境
```

### 各ファイルの説明

| ファイル名 | 説明 |
|-----------|------|
| `main.ipynb` | **メインファイル** - 完全な予測システムの実装 |
| `horse_racing_ai_prediction.ipynb` | 予測モデルの開発・検証用 |
| `horse_racing_ai_prediction_real_data.ipynb` | 実際のnetkeiba.comデータでの検証 |
| `horse_racing_ai_user_guide.md` | **詳細な使用方法** - 実用的な活用パターンを収録 |

---

## 🎯 主要な使用パターン

### パターン1: 📱 当日朝の一括チェック

```python
# 今日の全レースを一括でチェック
today_races = scraper.get_race_calendar()
for race in today_races:
    results = predict_specific_race(race['race_id'], threshold=0.3)
    if results is not None:
        recommended = results[results['recommendation'] == '◎']
        if len(recommended) > 0:
            print(f"🎯 {race['venue']} {race['race_name']}: {len(recommended)}頭推奨")
```

### パターン2: 🎯 特定レースの詳細分析

```python
# 特定レースの詳細分析
race_id = "202406050101"
results = predict_specific_race(race_id, threshold=0.3)

# 投資シミュレーション
recommended = results[results['recommendation'] == '◎']
total_investment = len(recommended) * 1000
expected_return = sum(recommended['prediction_proba'] * recommended['単勝'] * 1000)
print(f"投資額: ¥{total_investment:,}, 期待収益: ¥{expected_return:,.0f}")
```

### パターン3: 📊 リスク管理による投資判断

```python
# 保守的な予測（高精度・低リスク）
conservative_results = predict_specific_race(race_id, threshold=0.5)

# 積極的な予測（中精度・高リターン）
aggressive_results = predict_specific_race(race_id, threshold=0.15)
```

---

## 🔧 技術仕様

### 使用技術

- **プログラミング言語**: Python 3.8+
- **機械学習**: LightGBM, scikit-learn
- **データ処理**: pandas, numpy
- **Webスクレイピング**: requests, BeautifulSoup4
- **可視化**: matplotlib, seaborn
- **開発環境**: Jupyter Notebook

### 主要アルゴリズム

- **予測モデル**: LightGBM（勾配ブースティング）
- **特徴量エンジニアリング**: 過去成績、騎手・調教師成績、レース条件など
- **投資戦略**: 確率ベースの期待値最適化

---

## 📖 詳細ドキュメント

システムの詳細な使用方法については、以下をご参照ください：

- **[📚 ユーザーガイド](horse_racing_ai_user_guide.md)** - 実用的な活用方法の詳細
- **[📓 メインノートブック](main.ipynb)** - 完全なシステム実装
- **[🔬 開発ノートブック](horse_racing_ai_prediction.ipynb)** - モデル開発プロセス

---

## ⚠️ 注意事項

### 免責事項

- 本システムは教育・研究目的で開発されています
- 投資判断は自己責任で行ってください
- 過去の成績は将来の結果を保証するものではありません

### 利用規約

- netkeiba.comの利用規約を遵守してください
- スクレイピング頻度は適切な間隔を保ってください
- 商用利用の場合は別途ライセンスが必要です

---

## 🚀 今後の拡張予定

- [ ] リアルタイム通知機能
- [ ] Web APIの提供
- [ ] 複数競馬サイトへの対応
- [ ] ディープラーニングモデルの統合
- [ ] 自動投票機能（JRA-VAN連携）

---

## 📞 サポート

- **バグ報告**: GitHubのIssueをご利用ください
- **機能要望**: Discussionsでお聞かせください
- **技術的質問**: ユーザーガイドを先にご確認ください

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご確認ください。

---

*Last updated: 2025年8月7日*
