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

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
ada = AdaBoostClassifier(n_estimators=75, learning_rate=1.5)
etc = ExtraTreesClassifier(n_jobs=-1, criterion="entropy", n_estimators=5)

if __name__ == '__main__':
    eclf = VotingClassifier(estimators=[('ada', ada), ('rfc', rfc), ('etc', etc)], voting='soft', weights=[2, 1, 3],
                            n_jobs=1)

    eclf.fit(X, y)
    print("training finished")

    # save model for later ensemble
    with open(r'../data/voting2.pkl', 'wb') as f:
        pickle.dump(eclf, f)
    print("model dumped")
