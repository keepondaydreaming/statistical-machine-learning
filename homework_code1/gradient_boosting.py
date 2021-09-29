import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

feature_encoder = OneHotEncoder()
target_encoder = LabelEncoder()

def train(X, y):
    encoded_features = feature_encoder.transform(X).toarray()
    encoded_target = target_encoder.transform(y.values.ravel())

    dtrain = xgb.DMatrix(encoded_features, label=encoded_target)
    param = {'max_depth':2, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric': 'logloss'}
    num_round = 5
    bst = xgb.train(param, dtrain, num_round)

    return bst

def test(bst, X, y):
    encoded_features = feature_encoder.transform(X).toarray()
    encoded_target = target_encoder.transform(y.values.ravel())

    dtest = xgb.DMatrix(encoded_features, label=encoded_target)
    preds = bst.predict(dtest)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    
    acc = accuracy_score(encoded_target, preds)
    return acc


if __name__ == '__main__':
    features = ["Outlook", "Temperature", "Humidity", "Wind"]
    target = ["PlayTennis"]
    df = pd.read_csv("train.csv")

    X = df[features]
    y = df[target]

    feature_encoder.fit(X)
    target_encoder.fit(['No', 'Yes'])

    bst = train(X, y)

    df_test = pd.read_csv('test.csv')
    X_test = df_test[features]
    y_test = df_test[target]

    acc = test(bst, X_test, y_test)
    print(acc)
