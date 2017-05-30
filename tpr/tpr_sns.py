import numpy as np
import pandas as pd
import seaborn as sns
from data.data import Data
import pickle

with open('..\data\processed_data.pickle', 'rb') as f:
    df = pickle.load(f)
name = "srv_serror_rate"
sns_plt = sns.distplot(np.log(df[df[name] > 0][name] + 1), kde=False)
# sns_plt = sns.kdeplot(np.log(df[df[name] > 0][name]+1))
# sns_plt = sns.distplot(df[name])
fig = sns_plt.get_figure()
fig.savefig(name + ".png")
