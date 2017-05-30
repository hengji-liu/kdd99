from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
# scoring function
import scoring.scoring as sc

with open('../data/processed_data.pickle', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/feat_names.pickle', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

y = df["attack_type"]
X = df[selected_feat_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
print("splitting finished")

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print(precision_score(y_test, y_pred, average="micro"))
print(recall_score(y_test, y_pred, average='micro'))

# map to class function
y_transform = sc.map_to_major_classes(y_pred)

# parameters = {'n_estimators': list(range(10, 50, 10)),
#               # 'criterion': ("gini", "entropy"),
#               # 'max_features': ("sqrt", "log2"),
#               }
#
# if __name__ == '__main__':
#     gscv = GridSearchCV(rfc, parameters, verbose=2, refit=True, cv=3, n_jobs=1)
#     gscv.fit(X_train, y_train)
#
#     print(gscv.best_params_, gscv.best_score_)
#     print(gscv.score(X_test, y_test))
