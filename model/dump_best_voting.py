import pickle
from sklearn.ensemble import AdaBoostClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]

# TODO: put the best paras learn from grid search
ada = AdaBoostClassifier()  # add best params config
lrc = LogisticRegression(n_jobs=1)  # add best params config
etc = ExtraTreesClassifier(n_jobs=-1)  # add best params config

eclf = VotingClassifier(estimators=[('ada', ada), ('lrc', lrc), ('etc', etc)],
                        voting='hard', n_jobs=-1)

eclf.fit(X, y)
print("training finished")

# save model for later ensemble
with open(r'../data/voting.pkl', 'wb') as f:
    pickle.dump(eclf, f)
print("model dumped")
