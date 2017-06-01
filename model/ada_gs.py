from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]

ada = AdaBoostClassifier()

# TODO choose parameters for training
parameters = {
    'learning_rate': (0.5, 1, 2),
    'n_estimators': (50, 75, 100, 125, 150),
}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    # n_jobs = 1 due to limitied memory
    # refit = False for the sake of clarity
    gscv = GridSearchCV(ada, parameters, scoring=scorer, cv=3, verbose=2, refit=False, n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")
    # 1, 75
