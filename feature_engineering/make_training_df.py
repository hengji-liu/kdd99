from data.data import Data
import feature_engineering.functions as fe
import pickle

df = Data("10pc").df
# sparse feature merge
df = fe.merge_sparse_feature(df)
# one hot encoding
df = fe.one_hot(df)
# y labels mapping
df = fe.map2major5(df)

# the percentage show in the link below is for the 10pc data
# http://cseweb.ucsd.edu/~elkan/clresults.html
# print(df[df['attack_type'] == 0].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 1].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 2].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 3].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 4].shape[0] / df.shape[0])

with open(r'../data/training_df.pkl', 'wb') as f:
    pickle.dump(df, f)

print("pre-processing finished")
