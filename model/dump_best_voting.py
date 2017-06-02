import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20, criterion="entropy", max_features="sqrt")
ada = AdaBoostClassifier(n_estimators=75, learning_rate=1)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=4, criterion="entropy", max_features="log2", min_samples_split=2)

if __name__ == '__main__':
    eclf = VotingClassifier(estimators=[('ada', ada), ('rfc', rfc), ('etc', etc)], voting='soft', weights=[3, 5, 4],
                            n_jobs=1)

    eclf.fit(X, y)
    print("training finished")

    # save model for later ensemble
    with open(r'../data/voting.pkl', 'wb') as f:
        pickle.dump(eclf, f)
    print("model dumped")
