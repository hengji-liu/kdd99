# kdd99
vicious ntwk conn recog, kdd cup yr 99,  ml course final proj

task description: http://kdd.ics.uci.edu/databases/kddcup99/task.html

data: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

competition results: http://cseweb.ucsd.edu/~elkan/clresults.html

Preliminary Plan:
1. EDA (Exploratory Data Analysis)
2. Feature Engineering
- Sparse feature merging and one hot encoding
- Feature selection (maybe use RF feature importance?)
3. Scoring
- Map specific classes into the five major classes (igit)
- Scoring using the weight matrix (hengji)
4. Model Training
- Random Forrest Classifier (hengji)
- XGBoost Tree Booster (hengji)
- Ada Boost Classifier (hengji)
- Extra Trees Classifier (igit)
- Gradient Boosting Classifier (igit)
- Keras DNN (hengji, if time allows) (https://github.com/fchollet/keras)
- Random Greedy Forest Classifier (hengji, if time allows) (https://github.com/fukatani/rgf_python)
5. Ensemble (simply voting. if time allows, stacking)


run **feature_engineering/do.py** to generate a pickle on disk, then train using the pickle
