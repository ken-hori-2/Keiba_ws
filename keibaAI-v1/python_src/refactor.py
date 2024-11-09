import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb

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
        with open(file_path, "w") as f:
            f.write(f"# {self.model_name} Evaluation Results\n\n")
            f.write(f"- AUC (Train): {train_score:.4f}\n")
            f.write(f"- AUC (Test): {test_score:.4f}\n")
            f.write(f"- Accuracy (Test): {accuracy_test:.4f}\n")

    def plot_roc_curve(self, save_path="roc_curve.png"):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_test)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker="o", color="b", label="ROC Curve")
        plt.title(f"{self.model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.close()


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
        plt.savefig(save_path)
        plt.close()
        return sorted_importances


# Usage
# Step 1: Data processing
processor = RaceDataProcessor('RaceResults_and_RaceInfo.pickle')
preprocessed_data = processor.preprocess_data()
train_data, test_data = processor.split_data()
X_train, y_train, X_test, y_test = processor.get_features_and_labels()

# Step 2: Model training and evaluation (Random Forest)
rf_params = {
    "min_samples_split": 500,
    "max_depth": None,
    "n_estimators": 60,
    "criterion": "entropy",
    "class_weight": "balanced",
    "random_state": 100,
}
rf_model = RandomForestClassifier(**rf_params)
rf_trainer = ModelTrainer(rf_model, X_train, y_train, X_test, y_test, model_name="Random Forest")
rf_trainer.train()
train_score, test_score, accuracy_test = rf_trainer.evaluate()
rf_trainer.save_results(train_score, test_score, accuracy_test, file_path="rf_results.md")
rf_trainer.plot_roc_curve(save_path="rf_roc_curve.png")

# Display feature importance for Random Forest
rf_importances = FeatureImportance.display_importance(rf_model, X_train.columns, save_path="rf_feature_importance.png")
print(rf_importances)

# Step 3: Model training and evaluation (LightGBM)
lgb_params = {
    "num_leaves": 4,
    "n_estimators": 80,
    "class_weight": "balanced",
    "random_state": 100,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_trainer = ModelTrainer(lgb_model, X_train, y_train, X_test, y_test, model_name="LightGBM")
lgb_trainer.train()
train_score, test_score, accuracy_test = lgb_trainer.evaluate()
lgb_trainer.save_results(train_score, test_score, accuracy_test, file_path="lgb_results.md")
lgb_trainer.plot_roc_curve(save_path="lgb_roc_curve.png")

# Display feature importance for LightGBM
lgb_importances = FeatureImportance.display_importance(lgb_model, X_train.columns, save_path="lgb_feature_importance.png")
print(lgb_importances)
