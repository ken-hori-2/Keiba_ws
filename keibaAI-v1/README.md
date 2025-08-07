# keibaAI v1

## 概要

このプロジェクトでは、レースデータを基に予測モデルをトレーニングし、複数の機械学習アルゴリズムを組み合わせてアンサンブル学習を行う。具体的には、**ランダムフォレスト** (Random Forest) と **LightGBM** (Light Gradient Boosting Machine) を使い、これらの予測結果を基に **メタモデル** を学習させる。最終的に、複数のモデルの評価結果（AUC、精度など）を可視化し、最も優れたモデルを選択する。

コードの全体的な動作や使い方を説明する。

## 今回の評価結果
* Ensemble Meta Model Evaluation Results
    - AUC (Train): 0.8705
    - AUC (Test): 0.8054
    - Accuracy (Test): 0.7970
    
* Summary of All Models
    * RandomForest
        - AUC (Test): 0.7978
        - Accuracy (Test): 0.7340
    * LightGBM
        - AUC (Test): 0.8050
        - Accuracy (Test): 0.7218
    * Ensemble Meta Model
        - AUC (Test): 0.8054
        - Accuracy (Test): 0.7970


## 必要なライブラリ

以下のライブラリが必要：

- `pandas`
- `numpy`
- `joblib`
- `matplotlib`
- `scikit-learn`
- `lightgbm`

<!-- これらのライブラリは、`requirements.txt` に記載しておくと便利です。 -->

## ファイル構成

- `main.py`: メインの実行スクリプト。データの前処理からモデルの学習、評価、保存までを行う。
- `results_data/`: モデルの評価結果（AUC、精度）や可視化されたグラフを保存するディレクトリ。
- `best_model/`: 学習したモデル（ランダムフォレスト、LightGBM、メタモデル）を保存するディレクトリ。
- `RaceResults_and_RaceInfo.pickle`: 入力データファイル（レース結果やレース情報が含まれている）。



<!-- # 評価結果とROC曲線が以下のフォルダに保存される： -->
```
.
└── keibaAI-v1
    ├── JupyterNotebook_src/ (not use)
    ├── Practice_src/ (not use)
    └── python_src/ (here)
        ├── latest_src
        │   ├── main.py
        │   ├── best_model/
        │   └── results_data/
        │       ├── rf_results.md
        │       ├── lgb_results.md
        │       ├── ensemble_results.md
        │       ├── rf_roc_curve.png
        │       ├── lgb_roc_curve.png
        │       ├── ensemble_roc_curve.png
        │       ├── rf_feature_importance.png
        │       ├── lgb_feature_importance.png
        │       └── ensemble_feature_importance.png
        └── RaceResults_and_RaceInfo.pickle (2019)
```


## 使用方法

### 1. データの前処理とモデルの学習

- `RaceResults_and_RaceInfo.pickle` ファイルにレース結果データを保存する。
- スクリプト `main.py` を実行すると、データの前処理、学習、評価が順番に実行される。

```bash
cd python_src/latest_src/
python main.py
```

### 2. モデルの学習
スクリプト内で、以下の2つのモデルを学習させる：

ランダムフォレスト (RandomForestClassifier)
LightGBM (LGBMClassifier)
これらのモデルは、X_train と y_train を使って学習され、モデル評価のためにテストデータ (X_test, y_test) を使用する。

### 3. メタモデルの作成
ランダムフォレストとLightGBMの出力を使って、ロジスティック回帰 をメタモデルとして訓練する。このメタモデルは、両方のモデルからの予測結果を基に最終的な予測を行う。

### 4. モデル評価
各モデル（ランダムフォレスト、LightGBM、メタモデル）の評価が行われ、評価指標（AUC、精度）が計算される。評価結果は results_data/ フォルダ内に保存され、model_results.md に記録される。

### 5. 可視化
ROC曲線が各モデルについて描画され、results_data/ に保存される。
特徴量重要度（上位20の特徴量）はバーグラフとして表示され、results_data/ に保存される。

**主な関数の説明**
* RaceDataProcessor
    * データのロード、前処理、学習データとテストデータへの分割を行う。
* ModelTrainer
    * モデルの学習、評価（AUC、精度）、ROC曲線の描画、およびモデルの保存を行う。
* MetaModelTrainer
    * ランダムフォレストとLightGBMの予測をメタモデルに渡し、最終的な予測を行うために学習する。
* plot_feature_importance
    * 特徴量の重要度を可視化する。上位20の特徴量を表示し、その結果を results_data/ に保存する。

**保存される結果**
* 評価結果
    * 各モデルのAUC、テスト精度、訓練精度などの結果が results_data/ フォルダ内の .md ファイルに保存される。
* ROC曲線
    * 各モデルのROC曲線が画像（PNG）として results_data/ に保存される。
* 特徴量重要度
    * 上位20の特徴量の重要度がバーグラフとして可視化され、画像（PNG）として保存される。

* 結果の可視化
    * 各モデルの評価結果は、results_data/ フォルダに以下のように保存される：

**保存結果一覧**
* rf_results.md: ランダムフォレストモデルの評価結果

* lgb_results.md: LightGBMモデルの評価結果

* ensemble_results.md: メタモデルの評価結果

* rf_roc_curve.png: ランダムフォレストのROC曲線

* lgb_roc_curve.png: LightGBMのROC曲線

* ensemble_roc_curve.png: メタモデルのROC曲線

* rf_feature_importance.png: ランダムフォレストの特徴量重要度

* lgb_feature_importance.png: LightGBMの特徴量重要度

* ensemble_feature_importance.png: メタモデルの特徴量重要度

**結果の解釈**
* AUC (Test) や Accuracy (Test) はモデルの性能を示し、モデルがどれだけ良い予測をしているかを測定する。

* ROC曲線は、モデルの真陽性率（TPR）と偽陽性率（FPR）を示し、モデルの分類性能を視覚的に確認できます。

* 特徴量重要度の可視化では、最も重要な特徴量を確認し、モデルがどの特徴を重視して予測を行っているかを把握できます。

# 今後の改善点
より多くのモデル（XGBoostやSVMなど）を追加して、アンサンブルモデルを改善する。
データの前処理や特徴量エンジニアリングをさらに洗練させることで、モデルの精度を向上を図る。
深層学習モデル（LSTM, CNNなど）を導入し、時系列データの分析を行う。
特徴量エンジニアリングを高度化し、より精度の高い予測モデルを構築する。

# ライセンス
MIT License







<!-- 今後の展望
深層学習モデル（LSTM, CNNなど）を導入し、時系列データの分析を行う。
特徴量エンジニアリングを高度化し、より精度の高い予測モデルを構築する。 -->