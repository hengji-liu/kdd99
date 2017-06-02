from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"].values
X = df[selected_feat_names].values

etc = ExtraTreesClassifier(n_jobs=-1)

parameters_etc = {
    # 'criterion': ("gini", "entropy"),
    'n_estimators': (3, 4, 5, 6, 7),
    # 'max_features': ("sqrt", "log2"),
    'min_samples_split': (2, 3, 4),
}

scorer = cbs.scorer(True)

if __name__ == '__main__':
    gscv = GridSearchCV(estimator=etc,
                        param_grid=parameters_etc,
                        scoring=scorer,
                        verbose=10,
                        refit=False,
                        cv=3,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.cv_results_)
    print(gscv.best_params_)
    print(gscv.best_score_)
    # no significant
