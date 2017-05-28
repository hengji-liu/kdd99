from data.data import Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# X = df[names[:40]]
# y = df["attack_type"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)
# print(precision_score(y_true=y_test, y_pred=y_pred))