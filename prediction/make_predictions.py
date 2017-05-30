import pickle
from scoring import cost_based_scoring as cbs

with open('../data/test_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
X = df[selected_feat_names]
y = df['attack_type']  # ground truth
print("data loaded")

# rf
with open('../data/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
y_rf = rf.predict(X)
with open('../data/yrf.pkl', 'wb') as f:
    pickle.dump(y_rf, f)
print("rf results:")
cbs.score(y, y_rf, True)

# TODO: other model
