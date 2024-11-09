import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import lightgbm as lgb
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# CNNモデルの定義
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        # 入力層の次元数を3次元（例えば、(batch_size, features, 1)）にする
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128 * (input_dim - 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, features) -> (batch_size, 1, features)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Create folders for results and models
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

    def save_results(self, train_score, test_score, accuracy_test):
        file_path = os.path.join("results_data", f"{self.model_name}_results.md")
        with open(file_path, "w") as f:
            f.write(f"## {self.model_name} Evaluation Results\n")
            f.write(f"- AUC (Train): {train_score:.4f}\n")
            f.write(f"- AUC (Test): {test_score:.4f}\n")
            f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n\n")

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_test)
        plt.plot(fpr, tpr, marker="o", label=f"{self.model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.title(f"{self.model_name} ROC Curve")
        save_path = os.path.join("results_data", f"{self.model_name}_roc_curve.png")
        plt.savefig(save_path)
        plt.close()

    def save_model(self):
        joblib.dump(self.model, os.path.join("best_model", f"{self.model_name}.joblib"))

# CNNを使ったモデルのトレーニング
class CNNTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, model_name="CNN Model"):
        self.model = model
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32)
        self.X_test = torch.tensor(X_test.values, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float32)
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train.to(self.device)).squeeze()
            loss = self.criterion(outputs, self.y_train.to(self.device))
            loss.backward()
            self.optimizer.step()
        return self.model

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            y_pred_test = self.model(self.X_test.to(self.device)).squeeze()
        test_score = roc_auc_score(self.y_test, y_pred_test.cpu().numpy())
        accuracy_test = accuracy_score(self.y_test, (y_pred_test.cpu().numpy() > 0.5).astype(int))
        return test_score, accuracy_test

    def save_model(self, path="best_model/cnn_model.pth"):
        torch.save(self.model.state_dict(), path)

# メタモデルのトレーニング
class MetaModelTrainer:
    def __init__(self, rf_model, lgb_model, cnn_model, X_train, y_train, X_test, y_test):
        self.rf_model = rf_model
        self.lgb_model = lgb_model
        self.cnn_model = cnn_model
        self.meta_model = LogisticRegression()
        
        # 各モデルの予測を取り出し、メタ特徴量として使用
        self.rf_train_pred = rf_model.predict_proba(X_train)[:, 1]
        self.lgb_train_pred = lgb_model.predict_proba(X_train)[:, 1]
        self.cnn_train_pred = cnn_model.predict_proba(X_train)[:, 1]
        self.rf_test_pred = rf_model.predict_proba(X_test)[:, 1]
        self.lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
        self.cnn_test_pred = cnn_model.predict_proba(X_test)[:, 1]

        # メタ特徴量
        self.meta_X_train = pd.DataFrame({
            'rf_pred': self.rf_train_pred, 
            'lgb_pred': self.lgb_train_pred,
            'cnn_pred': self.cnn_train_pred
        })
        self.meta_X_test = pd.DataFrame({
            'rf_pred': self.rf_test_pred, 
            'lgb_pred': self.lgb_test_pred,
            'cnn_pred': self.cnn_test_pred
        })
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self):
        # self.meta_model.fit(self.meta_X_train, self.y_train)
        # meta_test_pred = self.meta_model.predict_proba(self.meta_X_test)[:, 1]
        
        # test_score = roc_auc_score(self.y_test, meta_test_pred)
        # accuracy_test = accuracy_score(self.y_test, (meta_test_pred > 0.5).astype(int))

        # # 結果保存
        # with open(os.path.join("results_data", "ensemble_results.md"), "a") as f:
        #     f.write(f"## Ensemble Meta Model Evaluation Results\n")
        #     f.write(f"- AUC (Test): {test_score:.4f}\n")
        #     f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n\n")

        # # ROC曲線を描画・保存
        # fpr, tpr, _ = roc_curve(self.y_test, meta_test_pred)
        # plt.plot(fpr, tpr, marker="o", label=f"Ensemble Model ROC Curve")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.legend()
        # plt.grid()
        # plt.savefig(os.path.join("results_data", "ensemble_roc_curve.png"))
        # plt.close()

        # # モデル保存
        # joblib.dump(self.meta_model, "best_model/meta_model.joblib")
        
        self.meta_model.fit(self.meta_X_train, self.y_train)
        meta_train_pred = self.meta_model.predict_proba(self.meta_X_train)[:, 1]
        meta_test_pred = self.meta_model.predict_proba(self.meta_X_test)[:, 1]
        train_score = roc_auc_score(self.y_train, meta_train_pred)
        test_score = roc_auc_score(self.y_test, meta_test_pred)
        accuracy_test = accuracy_score(self.y_test, (meta_test_pred > 0.5).astype(int))

        # Save ensemble results
        with open(os.path.join("results_data", "final_results.md"), "a") as f:
            f.write(f"## {self.model_name} Evaluation Results\n")
            f.write(f"- AUC (Train): {train_score:.4f}\n")
            f.write(f"- AUC (Test): {test_score:.4f}\n")
            f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n\n")

        # Plot and save ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, meta_test_pred)
        plt.plot(fpr, tpr, marker="o", label=f"{self.model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.title(f"{self.model_name} ROC Curve")
        plt.savefig(os.path.join("results_data", "ensemble_roc_curve.png"))
        plt.close()

        # Save ensemble model
        joblib.dump(self.meta_model, os.path.join("best_model", "meta_model.joblib"))

        # return test_score, accuracy_test
        return train_score, test_score, accuracy_test

class FeatureImportance:
    @staticmethod
    def display_importance(model, feature_names, top_n=20, save_path="feature_importance.png"):
        importances = pd.DataFrame({
            "features": feature_names,
            "importance": model.feature_importances_
        })
        sorted_importances = importances.sort_values("importance", ascending=False)[:top_n]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_importances["features"], sorted_importances["importance"], color="teal")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Top 20 Feature Importances")
        plt.gca().invert_yaxis()
        plt.grid()
        save_path = os.path.join("results_data", f"{save_path}")
        plt.savefig(save_path)
        plt.close()
        return sorted_importances


if __name__ == "__main__":
    # Process data
    processor = RaceDataProcessor('../../RaceResults_and_RaceInfo.pickle')
    processor.preprocess_data()
    processor.split_data()
    X_train, y_train, X_test, y_test = processor.get_features_and_labels()

    # Train Random Forest
    rf_model = RandomForestClassifier(
        min_samples_split=500, max_depth=None, n_estimators=60,
        criterion="entropy", class_weight="balanced", random_state=100
    )
    rf_trainer = ModelTrainer(rf_model, X_train, y_train, X_test, y_test, "RandomForest")
    rf_trainer.train()
    rf_train_score, rf_test_score, rf_accuracy_test = rf_trainer.evaluate()
    rf_trainer.save_results(rf_train_score, rf_test_score, rf_accuracy_test)
    rf_trainer.plot_roc_curve()
        # Save Random Forest model
    rf_trainer.save_model()
    
    #####
    # Display feature importance for LightGBM
    rf_importances = FeatureImportance.display_importance(rf_model, X_train.columns, save_path="rf_feature_importance.png")
    # print(rf_importances)
    #####

    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=100,
        learning_rate=0.05, num_leaves=32, random_state=100
    )
    lgb_trainer = ModelTrainer(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
    lgb_trainer.train()
    lgb_train_score, lgb_test_score, lgb_accuracy_test = lgb_trainer.evaluate()
    lgb_trainer.save_results(lgb_train_score, lgb_test_score, lgb_accuracy_test)
    lgb_trainer.plot_roc_curve()
    lgb_trainer.save_model()
    
    #####
    # Display feature importance for LightGBM
    lgb_importances = FeatureImportance.display_importance(lgb_model, X_train.columns, save_path="lgb_feature_importance.png")
    # print(lgb_importances)
    #####

    # # Train Meta Model (Ensemble)
    # meta_trainer = MetaModelTrainer(rf_model, lgb_model, X_train, y_train, X_test, y_test)
    # meta_train_score, meta_test_score, meta_accuracy_test = meta_trainer.train_and_evaluate()
    
    
    
    
    # CNNモデル
    cnn_model = CNNModel(input_dim=X_train.shape[1])  # 特徴量の次元数に合わせる
    cnn_trainer = CNNTrainer(cnn_model, X_train, y_train, X_test, y_test, "CNN Model")
    cnn_trainer.train()

    # メタモデル（アンサンブル）
    meta_model_trainer = MetaModelTrainer(rf_model, lgb_model, cnn_model, X_train, y_train, X_test, y_test)
    meta_train_score, meta_test_score, meta_accuracy_test = meta_model_trainer.train_and_evaluate()

    # Append final results summary
    with open(os.path.join("results_data", "final_results.md"), "a") as f:
        f.write("## Summary of All Models\n")
        f.write("### RandomForest\n")
        f.write(f"- AUC (Test): {rf_test_score:.4f}\n")
        f.write(f"- Accuracy (Test): {rf_accuracy_test:.4f}\n")
        f.write("### LightGBM\n")
        f.write(f"- AUC (Test): {lgb_test_score:.4f}\n")
        f.write(f"- Accuracy (Test): {lgb_accuracy_test:.4f}\n")
        f.write("### Ensemble Meta Model\n")
        f.write(f"- AUC (Test): {meta_test_score:.4f}\n")
        f.write(f"- Accuracy (Test): {meta_accuracy_test:.4f}\n")

    # Display -> Class()

    
    print("Training and evaluation completed. Results and models saved in 'results_data' and 'best_model' folders.")