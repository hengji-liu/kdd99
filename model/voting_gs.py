from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scoring import cost_based_scoring as cbs
import pickle

# Use this file to tune hyperparameters for voting classifier, ignore otherwise

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"].values
X = df[selected_feat_names].values

# plug optimum params for each classifier, or play around with parameters
# dict to fine tune to the voting classifier
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20, criterion="entropy", max_features="sqrt")
# ada = AdaBoostClassifier(n_estimators=75, learning_rate=1)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=4, criterion="entropy", max_features="log2", min_samples_split=2)

eclf = VotingClassifier(estimators=[('rfc', rfc), ('etc', etc)], n_jobs=1)

parameters = {  # 'classfier_name[double_underscores]key'
    # 'lrc__C': [1.0, 100.0],
    # 'etc__n_estimators': [20, 200],
    # 'voting': 'hard',
    # If we already have the optimal config or decide not to
    # fine tune each classifier individually, then
    # probably try to use 'soft' voting method. However, to use soft
    # voting, the weights for each classifier need to setup
    # correctly. Then find the right weight combination will be the
    # next target.
    'voting': ('soft',),
    'weights': [[1, 2],
                [2, 1],
                [1, 1],
                [3, 2],
                [2, 3]
                ]
    # 'weights': [[1, 2, 1],
    #             [2, 1, 1],
    #             [1, 1, 2],
    #             [1, 2, 3],
    #             [3, 2, 1],
    #             [2, 1, 3],
    #             [1, 1, 1],
    #             [2, 3, 1],
    #             [1, 3, 2],
    #             [3, 1, 2], ]
}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    # n_jobs = 1 due to limitied memory
    # refit = False for the sake of clarity
    gscv = GridSearchCV(eclf, parameters,
                        scoring=scorer,
                        cv=3,
                        verbose=10,
                        refit=True,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")

    # save model for later predicting
    with open(r'../data/stacking.pkl', 'wb') as f:
        pickle.dump(gscv, f)
    print("model dumped")
