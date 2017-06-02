from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from scoring import cost_based_scoring as cbs
import pickle

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
y = df["attack_type"].astype(int).values
X = df[selected_feat_names].values
print("data loaded")

# plug optimum params for each classifier
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20, criterion="entropy", max_features="sqrt")
ada = AdaBoostClassifier(n_estimators=75, learning_rate=1)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=4, criterion="entropy", max_features="log2", min_samples_split=2)
lr = LogisticRegression(n_jobs=-1)

# sclf = StackingCVClassifier(classifiers=[ada, rfc, etc], meta_classifier=lr, verbose=3, n_folds=3)
sclf = StackingClassifier(classifiers=[ada, rfc, etc], meta_classifier=lr, verbose=3)

parameters = {  # 'lower_case_classfier_class_name[double_underscores]key'
    'meta-logisticregression__C': (1, 5, 10, 50, 100)}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    # n_jobs = 1 due to limitied memory
    # refit = False for the sake of clarity
    gscv = GridSearchCV(sclf, parameters,
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
