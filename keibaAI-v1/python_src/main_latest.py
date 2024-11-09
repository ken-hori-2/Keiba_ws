import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import lightgbm as lgb

# フォルダ作成
os.makedirs('results_data', exist_ok=True)
os.makedirs('best_model', exist_ok=True)

class RaceDataProcessor:
    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        self.results = self._load_data()
        self.preprocessed_data = None
        self.train_data = None
        self.test_data = None

    def _load_data(self):
        return pd.read_pickle(self.pickle_path)

    def preprocess_data(self):
        df = self.results.copy()
        df = df[~(df["着順"].astype(str).str.contains(r"\D"))]
        df["着順"] = df["着順"].astype(int)

        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        df["単勝"] = df["単勝"].astype(float)
        df.drop(["タイム", "着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)
        
        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

        df['rank'] = df['着順'].map(lambda x: 1 if int(x) < 4 else 0)
        df.drop(["着順", "馬名"], axis=1, inplace=True)

        self.preprocessed_data = pd.get_dummies(df)
        return self.preprocessed_data

    def split_data(self, test_size=0.3):
        if self.preprocessed_data is None:
            raise ValueError("Data must be preprocessed before splitting.")

        sorted_id_list = self.preprocessed_data.sort_values("date").index.unique()
        train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
        test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
        
        self.train_data = self.preprocessed_data.loc[train_id_list]
        self.test_data = self.preprocessed_data.loc[test_id_list]
        return self.train_data, self.test_data

    def get_features_and_labels(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data must be split before extracting features and labels.")

        X_train = self.train_data.drop(["date", "rank"], axis=1)
        y_train = self.train_data["rank"]
        X_test = self.test_data.drop(["date", "rank"], axis=1)
        y_test = self.test_data["rank"]
        return X_train, y_train, X_test, y_test


class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, model_name="Model"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred_train = None
        self.y_pred_test = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred_train = self.model.predict_proba(self.X_train)[:, 1]
        self.y_pred_test = self.model.predict_proba(self.X_test)[:, 1]
        return self.y_pred_train, self.y_pred_test

    def evaluate(self):
        train_score = roc_auc_score(self.y_train, self.y_pred_train)
        test_score = roc_auc_score(self.y_test, self.y_pred_test)
        accuracy_test = accuracy_score(self.y_test, (self.y_pred_test > 0.5).astype(int))
        return train_score, test_score, accuracy_test

    def save_results(self, train_score, test_score, accuracy_test, file_path="model_results.md"):
        with open(os.path.join("results_data", file_path), "a") as f:
            f.write(f"## {self.model_name} Evaluation Results\n")
            f.write(f"- AUC (Train): {train_score:.4f}\n")
            f.write(f"- AUC (Test): {test_score:.4f}\n")
            f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n\n")

    def plot_roc_curve(self, save_path="roc_curve.png"):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_test)
        plt.plot(fpr, tpr, marker="o", label=f"{self.model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("results_data", save_path))
        plt.close()

    def save_model(self, path="best_model/model.joblib"):
        joblib.dump(self.model, path)


class MetaModelTrainer:
    def __init__(self, rf_model, lgb_model, X_train, y_train, X_test, y_test):
        # 各モデルの予測を特徴量にしてメタモデルをトレーニング
        self.rf_model = rf_model
        self.lgb_model = lgb_model
        self.meta_model = LogisticRegression()
        self.model_name = "Ensemble Meta Model"
        
        # ランダムフォレストとLightGBMの予測結果をメタモデルの入力として使用
        self.rf_train_pred = rf_model.predict_proba(X_train)[:, 1]
        self.lgb_train_pred = lgb_model.predict_proba(X_train)[:, 1]
        self.rf_test_pred = rf_model.predict_proba(X_test)[:, 1]
        self.lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
        
        self.meta_X_train = pd.DataFrame({'rf_pred': self.rf_train_pred, 'lgb_pred': self.lgb_train_pred})
        self.meta_X_test = pd.DataFrame({'rf_pred': self.rf_test_pred, 'lgb_pred': self.lgb_test_pred})
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self):
        self.meta_model.fit(self.meta_X_train, self.y_train)
        meta_train_pred = self.meta_model.predict_proba(self.meta_X_train)[:, 1]
        meta_test_pred = self.meta_model.predict_proba(self.meta_X_test)[:, 1]
        
        train_score = roc_auc_score(self.y_train, meta_train_pred)
        test_score = roc_auc_score(self.y_test, meta_test_pred)
        accuracy_test = accuracy_score(self.y_test, (meta_test_pred > 0.5).astype(int))

        # 結果保存
        with open(os.path.join("results_data", "ensemble_results.md"), "a") as f:
            f.write(f"## {self.model_name} Evaluation Results\n")
            f.write(f"- AUC (Train): {train_score:.4f}\n")
            f.write(f"- AUC (Test): {test_score:.4f}\n")
            f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n\n")

        # ROC曲線を描画・保存
        fpr, tpr, _ = roc_curve(self.y_test, meta_test_pred)
        plt.plot(fpr, tpr, marker="o", label=f"{self.model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("results_data", "ensemble_roc_curve.png"))
        plt.close()

        # モデル保存
        joblib.dump(self.meta_model, "best_model/meta_model.joblib")

        return train_score, test_score, accuracy_test

if __name__ == "__main__":
    # Usage
    processor = RaceDataProcessor('RaceResults_and_RaceInfo.pickle')
    processor.preprocess_data()
    processor.split_data()
    X_train, y_train, X_test, y_test = processor.get_features_and_labels()

    # ランダムフォレスト
    rf_model = RandomForestClassifier(**{
        "min_samples_split": 500, "max_depth": None, "n_estimators": 60,
        "criterion": "entropy", "class_weight": "balanced", "random_state": 100
    })
    rf_trainer = ModelTrainer(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    rf_trainer.train()
    rf_train_score, rf_test_score, rf_accuracy_test = rf_trainer.evaluate()
    rf_trainer.save_results(rf_train_score, rf_test_score, rf_accuracy_test, "rf_results.md")
    rf_trainer.plot_roc_curve("rf_roc_curve.png")
    rf_trainer.save_model("best_model/random_forest.joblib")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**{
        "n_estimators": 100, "random_state": 100, "class_weight": "balanced"
    })
    lgb_trainer = ModelTrainer(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
    lgb_trainer.train()
    lgb_train_score, lgb_test_score, lgb_accuracy_test = lgb_trainer.evaluate()
    lgb_trainer.save_results(lgb_train_score, lgb_test_score, lgb_accuracy_test, "lgb_results.md")
    lgb_trainer.plot_roc_curve("lgb_roc_curve.png")
    lgb_trainer.save_model("best_model/lgbm.joblib")

    # メタモデル（アンサンブル）
    meta_trainer = MetaModelTrainer(rf_model, lgb_model, X_train, y_train, X_test, y_test)
    meta_train_score, meta_test_score, meta_accuracy_test = meta_trainer.train_and_evaluate()