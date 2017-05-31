from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import voting_classifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
# print("splitting finished")

rfc = RandomForestClassifier(n_jobs=-1)
etc = ExtraTreesClassifier(n_jobs=-1)
gbc = GradientBoostingClassifier()
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)
# print("training finished")
# print(cbs.score(y_test, y_pred, True))

# TODO choose parameters for training
parameters_rfc = {
    # 'n_estimators': list(range(10, 100, 10)),
    'criterion': ("gini", "entropy"),
    # 'max_features': ("sqrt", "log2"),
}

parameters_etc = {
    # 'n_estimators': list(range(10, 100, 10)),
    'criterion': ("gini", "entropy"),
    # 'max_features': ("sqrt", "log2"),
}

parameters_gbc = {
    # loss : {‘deviance’, ‘exponential’},
    # learning_rate : float, optional (default=0.1),
    # n_estimators : int (default=100),
    # max_depth : integer, optional (default=3),
    'criterion': ('friedman_mse', 'mae'),
    'n_estimators': (500,),
    # 'max_features': ("sqrt", "log2"),
}

scorer = cbs.scorer(True)

if __name__ == '__main__':
    # gscv = GridSearchCV(rfc,
    #                     parameters_rfc,
    #                     scoring=scorer,
    #                     verbose=2,
    #                     refit=True,
    #                     cv=3,
    #                     n_jobs=-1)
    # gscv.fit(X, y)
    # print(gscv.best_params_, gscv.best_score_)

    # gscv = GridSearchCV(etc,
    #                     parameters_etc,
    #                     scoring=scorer,
    #                     verbose=2,
    #                     refit=True,
    #                     cv=3,
    #                     n_jobs=-1)
    # gscv.fit(X, y)
    # print(gscv.best_params_, gscv.best_score_)

    gscv = GridSearchCV(gbc,
                        parameters_gbc,
                        scoring=scorer,
                        verbose=2,
                        refit=True,
                        # cv=2,
                        n_jobs=1)
    gscv.fit(X, y)
    print(gscv.best_params_, gscv.best_score_)

    # TODO: save model for later ensemble voting
