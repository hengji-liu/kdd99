from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from scoring import cost_based_scoring as cbs
import pickle

# Use this file to tune hyperparameters for stacking classifier,
# ignore otherwise

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]

# plug optimum params for each classifier, or play around with parameters
# dict to fine tune to the voting classifier
ada = AdaBoostClassifier()
rfc = RandomForestClassifier(n_jobs=-11)
etc = ExtraTreesClassifier(n_jobs=-1)
lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[ada, rfc, etc],
                          meta_classifier=lr)


parameters = {  # 'lower_case_classfier_class_name[double_underscores]key'
              'randomforestclassifier__n_estimators': [20, 200],}

scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    # n_jobs = 1 due to limitied memory
    # refit = False for the sake of clarity
    gscv = GridSearchCV(sclf, parameters,
                        scoring=scorer,
                        cv=3,
                        verbose=10,
                        refit=True,
                        n_jobs=1)
    gscv.fit(X, y)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")
