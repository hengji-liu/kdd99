import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

with open('../data/processed_data.pickle', 'rb') as f:
    df = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df.drop("attack_type", axis=1)
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X, y)
print("training finished")

importances = rfc.feature_importances_
indices = np.argsort(importances)
selected_feat_names = []
for f in range(X.shape[1]):
    if importances[indices[f]] != 0:
    # if importances[indices[f]] - 0.000001 >= 0:
        selected_feat_names.append(X.columns[f])
    print("%2d) %-*s %f" % (f + 1, 30, X.columns[f], importances[indices[f]]))

print(len(selected_feat_names), "features are selected")

with open(r'..\data\feat_names.pickle', 'wb') as f:
    pickle.dump(selected_feat_names, f)
