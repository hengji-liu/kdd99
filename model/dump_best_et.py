import pickle
from sklearn.ensemble import ExtraTreesClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]

etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=4, criterion="entropy", max_features="log2", min_samples_split=2)
etc.fit(X, y)
print("training finished")

# save model for later ensemble
with open(r'../data/et.pkl', 'wb') as f:
    pickle.dump(etc, f)
print("model dumped")
