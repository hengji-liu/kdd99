from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]

lrc = LogisticRegression(n_jobs=1)

parameters_lrc = {
    'multi_class': ('ovr', 'multinomial'),
    'solver': ('newton-cg', 'lbfgs'),
    # not liblinear because dataset large and is multiclass
    # not sag because feature not same scale
    'max_iter': (100, 200,),
    # 'random_state': (10,),
    'tol': (1e-4, 1e-5),
    'C': (1.0, 5.0, 10.0, 50.0, 100)
}

scorer = cbs.scorer(True)

if __name__ == '__main__':
    gscv = GridSearchCV(estimator=lrc,
                        param_grid=parameters_lrc,
                        scoring=scorer,
                        verbose=10,
                        refit=False,
                        cv=3,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.best_params_)
    print(gscv.best_score_)
