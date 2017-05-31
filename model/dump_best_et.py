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
# TODO: put the best paras learn from grid search
etc = ExtraTreesClassifier(n_jobs=-1, ) # fill it with the result from et_gs
etc.fit(X, y)
print("training finished")

# save model for later ensemble
with open(r'../data/et.pkl', 'wb') as f:
    pickle.dump(etc, f)
print("model dumped")
