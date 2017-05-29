import numpy as np
import pandas as pd
import seaborn as sns
from data.data import Data

df = Data('full').df
# sns.set_style("whitegrid")
# sns.countplot(y=df["attack_type"])
# sns.distplot(np.log(df[df['duration'] > 0]['duration']))
sns.distplot(df['duration'])
# sns.factorplot("attack_type", col="service", data=df, kind="count", col_wrap=8, size=3, aspect=.8)
sns.plt.show()
