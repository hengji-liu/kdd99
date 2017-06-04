from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
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
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
ada = AdaBoostClassifier(n_estimators=75, learning_rate=1.5)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=5, criterion="entropy")

lr = LogisticRegression(n_jobs=-1)

sclf = StackingCVClassifier(classifiers=[rfc, etc], meta_classifier=lr, use_probas=True, verbose=3, n_folds=3)
# sclf = StackingClassifier(classifiers=[ada, rfc, etc], meta_classifier=lr, use_probas=True, verbose=3)

parameters = {  # 'lower_case_classfier_class_name[double_underscores]key'
    # 'meta-logisticregression__C': (1, 5, 10, 50, 100, 500, 1000)}
    'meta-logisticregression__C': (1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   10, 20, 30, 40, 50, 60, 70, 80, 90,
                                   100, 200, 300, 400, 500, 600, 700, 800, 900,
                                   1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)}

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
    print(gscv.cv_results_)
    print("grid search finished")
    # 'mean_test_score': array([-0.04454018, -0.04659451, -0.04870008, -0.05587646, -0.04658757, -0.0459294 , -0.05374047])

# 'rank_test_score': array([21, 9, 30, 7, 12, 15, 32, 1, 23,
#                           14, 10, 8, 24, 19, 34, 18, 2, 28,
#                           25, 3, 17, 37, 13, 20, 27, 33, 16,
#                           4, 26, 29, 22, 35, 31, 11, 6, 36, 5])
#  'mean_test_score': array([-0.04687664, -0.0451367 , -0.04993048, -0.04511649, -0.04539168, -0.04561359, -0.05266788, -0.04360212, -0.04699586,
#                            -0.04556194, -0.04522611, -0.04512241, -0.04706936, -0.04607761, -0.05465546, -0.04607251, -0.04372604, -0.04800374,
#                            -0.04718633, -0.04377095, -0.04579589, -0.05831132, -0.04547048, -0.04610701, -0.0472339 , -0.05281569, -0.04577139,
#                            -0.04430276, -0.04720777, -0.048558  , -0.04694197, -0.0548545 , -0.0511574 , -0.04534268, -0.04486824, -0.05485471, -0.04430276])
# seems to be very random