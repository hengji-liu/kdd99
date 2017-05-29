from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pickle

with open('..\data\processed_data.pickle', 'rb') as f:
    df = pickle.load(f)

# TODO: cross validation, grid search and stuff
y = df["attack_type"]
X = df.drop("attack_type", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
print("splitting finished")

rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("training finished")

# TODO: find more related feature and retrain
print("feature importance: ", rfc.feature_importances_)

# TODO: more specific means measure a multi-class classification
print("precision: ", precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
