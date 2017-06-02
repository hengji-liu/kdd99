import pickle
from scoring import cost_based_scoring as cbs

with open('../data/test_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
X = df[selected_feat_names].values
y = df['attack_type'].values  # ground truth
print("data loaded")

# rf
with open('../data/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
y_rf = rf.predict(X)
print("rf results:")
cbs.score(y, y_rf, True)

# ada boost
with open('../data/ada.pkl', 'rb') as f:
    ada = pickle.load(f)
y_ada = ada.predict(X)
print("ada results:")
cbs.score(y, y_ada, True)

# et
with open('../data/et.pkl', 'rb') as f:
    et = pickle.load(f)
y_et = et.predict(X)
print("et results:")
cbs.score(y, y_et, True)

# vt
with open('../data/voting.pkl', 'rb') as f:
    voting = pickle.load(f)
y_voting = voting.predict(X)
print("voting results:")
cbs.score(y, y_voting, True)

# stacking
with open('../data/stacking.pkl', 'rb') as f:
    stacking = pickle.load(f)
y_stacking = stacking.predict(X)
print("stacking results:")
cbs.score(y, y_stacking, True)
