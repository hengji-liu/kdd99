import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df.drop("attack_type", axis=1)

"""
After several experiments, it is indicated that 40+ features has major importance to this task
Select features that appear in the top 50 in all 10 trainings will yield 40+ features
"""
# first training
selected_feat_names = set()
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X, y)
print("training finished")

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]  # descending order
for f in range(X.shape[1]):
    if f < 50:
        selected_feat_names.add(X.columns[indices[f]])
    print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))

# plt.title("Feature Importance")
# plt.bar(range(X.shape[1]),
#         importances[indices],
#         color='lightblue',
#         align='center')
# plt.xticks(range(X.shape[1]),
#            X.columns[indices],
#            rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.tight_layout()
# plt.show()

for i in range(9):
    tmp = set()
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(X, y)
    print("training finished")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]  # descending order
    for f in range(X.shape[1]):
        if f < 50:  # need roughly more than 40 features according to experiments
            tmp.add(X.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))

    selected_feat_names &= tmp
    print(len(selected_feat_names), "features are selected")

with open(r'../data/selected_feat_names.pkl', 'wb') as f:
    pickle.dump(list(selected_feat_names), f)
