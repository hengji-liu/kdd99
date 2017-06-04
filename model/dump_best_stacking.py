import pickle
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"].values
X = df[selected_feat_names].values

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
ada = AdaBoostClassifier(n_estimators=75, learning_rate=1.5)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=5, criterion="entropy")
# lr = LogisticRegression(n_jobs=-1, C=100)  # meta classifier, 2 trees, c=100 is used in stacking2.pkl
lr = LogisticRegression(n_jobs=-1, C=8)  # meta classifier

sclf = StackingCVClassifier(classifiers=[ada, rfc, etc], meta_classifier=lr, use_probas=True, n_folds=3, verbose=3)
# sclf = StackingCVClassifier(classifiers=[rfc, etc], meta_classifier=lr, use_probas=True, n_folds=3, verbose=3)

sclf.fit(X, y)
print("training finished")

# save model for later predicting
with open(r'../data/stacking.pkl', 'wb') as f:
    pickle.dump(sclf, f)
print("model dumped")
