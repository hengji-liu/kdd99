import numpy as np
import pandas as pd
import seaborn as sns
from data.data import Data
import pickle

# with open(r'../data/training_df.pkl', 'rb') as f:
#     df = pickle.load(f)

df = Data('full').df
name = "is_host_login"
# sns_plt = sns.countplot(df["attack_type"])
# sns_plt = sns.countplot(df["protocol_type"])
# sns_plt = sns.countplot(y=df["service"])
# sns_plt = sns.countplot(x=df["su_attempted"])
sns_plt = sns.countplot(x=df["is_host_login"])
# sns_plt = sns.distplot(df["duration"], kde=False)
# sns_plt = sns.distplot(df["num_root"], kde=False)
# sns.plt.show()
fig = sns_plt.get_figure()
fig.savefig(name + ".png")
