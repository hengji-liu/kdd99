import pickle
from sklearn.ensemble import AdaBoostClassifier, \
    ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]

# TODO: put the best param learn from grid search
ada = AdaBoostClassifier()  # add best params config
etc = ExtraTreesClassifier(n_jobs=-1)  # add best params config
rfc = RandomForestClassifier(n_jobs=-11)

lr = LogisticRegression()  # meta classifier

sclf = StackingClassifier(classifiers=[ada, rfc, etc],
                          meta_classifier=lr)

sclf.fit(X, y)
print("training finished")

# save model for later ensemble
with open(r'../data/stacking.pkl', 'wb') as f:
    pickle.dump(sclf, f)
print("model dumped")
