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

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20, criterion="entropy", max_features="sqrt")
# ada = AdaBoostClassifier(n_estimators=75, learning_rate=1)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=4, criterion="entropy", max_features="log2", min_samples_split=2)
lr = LogisticRegression(n_jobs=-1, C=1000)  # meta classifier

sclf = StackingClassifier(classifiers=[rfc, etc],
                          meta_classifier=lr)

sclf.fit(X, y)
print("training finished")

# save model for later predicting
with open(r'../data/stacking.pkl', 'wb') as f:
    pickle.dump(sclf, f)
print("model dumped")
