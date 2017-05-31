from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]

etc = ExtraTreesClassifier(n_jobs=-1)

# TODO choose parameters for training
parameters_etc = {
    # to make sure each iteration of cv yield to same random number, so the
    # result of cv can be compared
    'random_state': (100,),
    'criterion': ("gini", "entropy"),
    'n_estimators': (10, 20),
    'max_features': ("sqrt", "log2"),
    'min_samples_split': (2, 3),
}

scorer = cbs.scorer(True)

if __name__ == '__main__':
    gscv = GridSearchCV(estimator=etc,
                        param_grid=parameters_etc,
                        scoring=scorer,
                        verbose=10,
                        refit=True,
                        cv=3,
                        n_jobs=1)
    gscv.fit(X, y)
    print(gscv.best_params_)
    print(gscv.best_score_)
