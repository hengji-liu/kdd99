from data.data import Data
from feature_engineering import merge_sparse_feature
from feature_engineering import one_hot
import pickle

df = Data("full").df
# sparse feature merge
df = merge_sparse_feature.transfrom(df)
# one hot encoding
df = one_hot.transform(df)

with open(r'../data/processed_data.pickle', 'wb') as f:
    pickle.dump(df, f)

print("pre-processing finished")
