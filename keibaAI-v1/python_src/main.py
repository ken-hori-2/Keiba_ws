import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
# from tqdm.notebook import tqdm
from tqdm import tqdm
import re


def scrape_race_data():
    # 2024/11/9はスクレイピングがエラーになるので、レース結果とレース情報のみ使用
    # resultsにレース結果とレース情報をマージ済み
    
    results = pd.read_pickle('RaceResults_and_RaceInfo.pickle')
    return results

results = scrape_race_data()
print(results)


# date列の処理を追加
def preprocessing(results):
    df = results.copy()

    # # 着順に数字以外の文字列が含まれているものを取り除く
    # df = df[~(df["着 順"].astype(str).str.contains("\D"))]
    # df["着順"] = df["着 順"].astype(int)
    # # df["着 順"] = df["着 順"].astype(int)
    # スペースを消せたので以下に変更 また、正規表現の前にrを追加
    df = df[~(df["着順"].astype(str).str.contains(r"\D"))]
    df["着順"] = df["着順"].astype(int)

    # 性齢を性と年齢に分ける
    df["性"] = df["性齢"].map(lambda x: str(x)[0])
    df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

    # 馬体重を体重と体重変化に分ける
    df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
    df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

    # データをint, floatに変換
    df["単勝"] = df["単勝"].astype(float)

    # 不要な列を削除
    df.drop(["タイム", "着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)
    
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

    return df



#前処理
results_p = preprocessing(results)

#着順を0or1にする
# results_p['rank'] = results_p['着 順'].map(lambda x: 1 if x<4 else 0)
results_p['rank'] = results_p['着順'].map(lambda x: 1 if int(x)<4 else 0)
results_p.drop(['着順'], axis=1, inplace=True) # 着順が消えない(半角スペースがあるため？)
# results_p.drop(['着 順'], axis=1, inplace=True)

#もし、動画のように着順をdropしてrankを作っている場合は
# results_p['rank'] = results_p['rank'].map(lambda x: 1 if x<4 else 0)

results_p.drop(["馬名"], axis=1, inplace=True)
results_d = pd.get_dummies(results_p)


results_p['rank'].value_counts()

#時系列に沿ってデータを分割
def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test

train, test = split_data(results_d, test_size=0.3)
X_train = train.drop(["date", "rank"], axis=1)
y_train = train["rank"]
X_test = test.drop(["date", "rank"], axis=1)
y_test = test["rank"]



#ランダムフォレストによる予測モデル作成
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=100)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:, 1]

print(y_pred)






#ROC曲線の表示
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# #jupyterlabを使う場合、この2行はいらない
# from jupyterthemes import jtplot
# jtplot.style(theme="monokai")

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, marker="o")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.grid()
plt.show()

#AUCスコアの表示
roc_auc_score(y_test, y_pred)
y_pred_train = rf.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_pred_train)

print(roc_auc_score(y_test, y_pred), roc_auc_score(y_train, y_pred_train))


# y_train # ['rank']
y_test[:20] # ['rank']



#パラメータの調整
params = {
    "min_samples_split": 500,
    "max_depth": None,
    "n_estimators": 60,
    "criterion": "entropy",
    "class_weight": "balanced",
    "random_state": 100,
}
rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)
y_pred_train = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_train, y_pred_train))
print(roc_auc_score(y_test, y_pred))

#変数の重要度の表示
importances = pd.DataFrame(
    {"features": X_train.columns, "importance": rf.feature_importances_}
)
importances.sort_values("importance", ascending=False)[:20]






#LightGBMによる予測モデル作成
import lightgbm as lgb

params = {
    "num_leaves": 4,
    "n_estimators": 80,
    #'min_data_in_leaf': 15,
    "class_weight": "balanced",
    "random_state": 100,
}
lgb_clf = lgb.LGBMClassifier(**params)
lgb_clf.fit(X_train.values, y_train.values)
y_pred_train = lgb_clf.predict_proba(X_train)[:, 1]
y_pred = lgb_clf.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_train, y_pred_train))
print(roc_auc_score(y_test, y_pred))

#変数の重要度の表示
importances = pd.DataFrame(
    {"features": X_train.columns, "importance": lgb_clf.feature_importances_}
)
importances.sort_values("importance", ascending=False)[:20]