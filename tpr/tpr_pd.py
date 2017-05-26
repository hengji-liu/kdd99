import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

file = r"..\data\train10pc"
names = ["duration",
         "protocol_type",
         "service",
         "flag",
         "src_bytes",
         "dst_bytes",
         "land",
         "wrong_fragment",
         "urgent",
         "hot",
         "num_failed_logins",
         "logged_in",
         "num_compromised",
         "root_shell",
         "su_attempted",
         "num_root",
         "num_file_creations",
         "num_shells",
         "num_access_files",
         "num_outbound_cmds",
         "is_host_login",
         "is_guest_login",
         "count",
         "srv_count",
         "serror_rate",
         "srv_serror_rate",
         "rerror_rate",
         "srv_rerror_rate",
         "same_srv_rate",
         "diff_srv_rate",
         "srv_diff_host_rate",
         "dst_host_count",
         "dst_host_srv_count",
         "dst_host_same_srv_rate",
         "dst_host_diff_srv_rate",
         "dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate",
         "dst_host_serror_rate",
         "dst_host_srv_serror_rate",
         "dst_host_rerror_rate",
         "dst_host_srv_rerror_rate",
         "attack_type"
         ]

df = pd.read_csv(file, header=None, names=names)
print(df.size)
print(df.shape)
print("---")
print(df.describe(include=[np.object_]))  # categorical features
print("---")
print(df.describe(include=[np.number]))  # numeric features
# https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
