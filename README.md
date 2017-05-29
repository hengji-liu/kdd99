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
- Random Forrest Classifier (hengji)
- Extra Trees Classifier (igit)
- Gradient Boosting Classifier (igit)
- XGBoost Linear Booster (hengji)
- XGBoost Tree Booster (hengji)
- Support Vector Classifier (igit)
- Ridge Classifier (hengji)
- Keras DNN (hengji, if time allows) (https://github.com/fchollet/keras)
- Random Greedy Forest Classifier (hengji, if time allows) (https://github.com/fukatani/rgf_python)
4. Scoring
- Map specific classes into the five major classes (igit)
- Scoring using the weight matrix (hengji)
5. Ensemble (simply voting. if time allows, stacking)


run **feature_engineering/do.py** to generate a pickle on disk, then train using the pickle
