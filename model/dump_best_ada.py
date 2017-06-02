import pickle
from sklearn.ensemble import AdaBoostClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]

ada = AdaBoostClassifier(n_estimators=75, learning_rate=1)
ada.fit(X, y)
print("training finished")

# save model for later ensemble
with open(r'../data/ada.pkl', 'wb') as f:
    pickle.dump(ada, f)
print("model dumped")
