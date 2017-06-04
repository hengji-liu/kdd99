import numpy as np
import pandas as pd
import seaborn as sns
from data.data import Data
from feature_engineering import feat_utils
import pickle

with open(r'../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)

# df = Data('full').df

'''
before feature engineering
'''
# sns_plt = sns.countplot(y=df["attack_type"])
# sns_plt = sns.distplot(df["rerror_rate"], kde=False)
# sns_plt = sns.countplot(df["protocol_type"])
# sns_plt = sns.countplot(y=df["service"])
# sns_plt = sns.countplot(x=df["su_attempted"])
# sns_plt = sns.countplot(x=df["is_host_login"])
# sns_plt = sns.distplot(df["duration"], kde=False)
# sns_plt = sns.distplot(df["num_root"], kde=False)

'''
during feature engineering
'''
# df = feat_utils.merge_sparse_feature(df)
# sns_plt = sns.countplot(y=df["service"]) #
# no big difference in pic
# sns_plt = sns.countplot(df["attack_type"])

print(df.loc[(df['attack_type'] == 0), 'attack_type'].size)
print(df.loc[(df['attack_type'] == 1), 'attack_type'].size)
print(df.loc[(df['attack_type'] == 2), 'attack_type'].size)
print(df.loc[(df['attack_type'] == 3), 'attack_type'].size)
print(df.loc[(df['attack_type'] == 4), 'attack_type'].size)

df.loc[(df['attack_type'] == 0), 'attack_type'] = '0 - normal'
df.loc[(df['attack_type'] == 1), 'attack_type'] = '1 - probe'
df.loc[(df['attack_type'] == 2), 'attack_type'] = '2 - denial of service (DOS)'
df.loc[(df['attack_type'] == 3), 'attack_type'] = '3 - user-to-root (U2R)'
df.loc[(df['attack_type'] == 4), 'attack_type'] = '4 - remote-to-local (R2L)'

sns_plt = sns.countplot(x=df["attack_type"],
                        order=['0 - normal', '1 - probe', '2 - denial of service (DOS)', '3 - user-to-root (U2R)',
                               '4 - remote-to-local (R2L)'])

'''
plot
'''
sns.plt.show()
# fig = sns_plt.get_figure()
# fig.savefig("attack_type_before_mapping" + ".png")
