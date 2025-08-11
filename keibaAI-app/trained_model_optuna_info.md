# Horse Racing AI Prediction System - 保存情報

## 保存日時
2025年08月07日 21:13:12

## システム状態
- 学習済み: はい
- データ読み込み済み: いいえ
- 評価器準備済み: いいえ

## 保存ファイル
- モデル: trained_model_optuna_model.pkl
- 処理済みデータ: trained_model_optuna_processed_data.pkl  
- システム状態: trained_model_optuna_system_state.json
- この情報: trained_model_optuna_info.md

## 使用方法
```python
# モデルの読み込み
predictor = HorseRacingPredictor()
success = predictor.load_model('trained_model_optuna.pkl')
if success:
    # 予測を実行
    results = predictor.predict_race(race_id)
```
