# kdd99
vicious ntwk conn recog, kdd cup yr 99,  ml course final proj

task description: http://kdd.ics.uci.edu/databases/kddcup99/task.html

data: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

Preliminary Plan:
1. EDA (Exploratory Data Analysis)
2. Feature Engineering
- Sparse feature merging and one hot encoding
- Feature selection (maybe use RF feature importance?)
3. Model Training
- Random Forrest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- XGBoost Linear Booster
- XGBoost Tree Booster
- Support Vector Classifier
- Ridge Classifier
- Keras DNN (https://github.com/fchollet/keras)
- Random Greedy Forest Classifier (https://github.com/fukatani/rgf_python)
4. Ensemble (simply voting. if time allows, stacking)
