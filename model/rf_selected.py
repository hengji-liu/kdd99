import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pickle

with open('..\data\processed_data.pickle', 'rb') as f:
    df = pickle.load(f)

with open(r'..\data\feat_names.pickle', 'rb') as f:
    selected_feat_names = pickle.load(f)

print(len(selected_feat_names))
# TODO: cross validation, grid search and stuff
y = df["attack_type"]
X = df[selected_feat_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
print("splitting finished")

rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy')
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("training finished")

print("precision: ", precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
