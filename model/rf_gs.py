from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]

rfc = RandomForestClassifier(n_jobs=-1)

# TODO choose parameters for training
parameters = {
    # 'n_estimators': list(range(10, 100, 10)),
    'criterion': ("gini", "entropy"),
    # 'max_features': ("sqrt", "log2"),
}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    gscv = GridSearchCV(rfc, parameters, scoring=scorer, verbose=2, refit=True, cv=3, n_jobs=1)
    gscv.fit(X, y)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")
