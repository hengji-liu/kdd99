from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"].values
X = df[selected_feat_names].values

rfc = RandomForestClassifier(n_jobs=-1)

parameters = {
    # 'n_estimators': tuple(range(10, 50, 10)),  # overfit if too large, underfit if too small
    'n_estimators': [25, 30, 35],
    'criterion': ("gini", "entropy")
}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    # n_jobs = 1 due to limitied memory
    gscv = GridSearchCV(rfc, parameters,
                        scoring=scorer,
                        cv=3,
                        verbose=2,
                        refit=False,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.cv_results_)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")
    # 35
