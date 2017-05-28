import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

########## can't pass in string param ##########
# TODO: one hot encoding? merge sparse categorical feature
X = [["zero", "zero"], ["one", "one"]]
Y = ["zero", "one"]
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
clf = clf.fit(X, Y)

file = r"..\data\train10pc"
names = ["duration",  # length (number of seconds) of the conn's
         "protocol_type",  # symbolic, type of the protocol, e.g. tcp, udp, etc.
         "service",  # symbolic, network service on the destination, e.g., http, telnet, etc.
         "flag",  # symbolic, normal or error status of the conn
         "src_bytes",  # number of data bytes from source to destination
         "dst_bytes",  # number of data bytes from destination to source
         "land",  # symbolic, 1 if conn is from/to the same host/port; 0 otherwise
         "wrong_fragment",  # number of ''wrong'' fragments 
         "urgent",  # number of urgent packets
         # ----------
         # ----- Basic features of individual TCP conn's -----
         # ----------
         "hot",  # number of ''hot'' indicators
         "num_failed_logins",  # number of failed login attempts 
         "logged_in",  # symbolic, 1 if successfully logged in; 0 otherwise
         "num_compromised",  # number of ''compromised'' conditions 
         "root_shell",  # 1 if root shell is obtained; 0 otherwise 
         "su_attempted",  # 1 if ''su root'' command attempted; 0 otherwise 
         "num_root",  # number of ''root'' accesses 
         "num_file_creations",  # number of file creation operations
         "num_shells",  # number of shell prompts 
         "num_access_files",  # number of operations on access control files
         "num_outbound_cmds",  # number of outbound commands in an ftp session 
         "is_host_login",  # symbolic, 1 if the login belongs to the ''hot'' list; 0 otherwise 
         "is_guest_login",  # symbolic, 1 if the login is a ''guest''login; 0 otherwise 
         # ----------
         # ----- Content features within a conn suggested by domain knowledge -----
         # ----------
         "count",  # number of conn's to the same host as the current conn in the past two seconds 
         # Time-based Traffic Features (examine only the conn in the past two seconds):
         # 1. Same Host, have the same destination host as the current conn
         # 2. Same Service, have the same service as the current conn.
         "srv_count",  # SH, number of conn's to the same service as the current conn
         "serror_rate",  # SH, % of conn's that have SYN errors
         "srv_serror_rate",  # SS, % of conn's that have SYN errors
         "rerror_rate",  # SH, % of conn's that have REJ errors 
         "srv_rerror_rate",  # SS, % of conn's that have REJ errors 
         "same_srv_rate",  # SH, % of conn's to the same service 
         "diff_srv_rate",  # SH, % of conn's to different services 
         "srv_diff_host_rate",  # SH,  % of conn's to different hosts 
         # ----------
         # Host-base Traffic Features, constructed using a window of 100 conn's to the same host
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
         # ----------
         # category
         "attack_type"
         ]
df = pd.read_csv(file, header=None, names=names)
X = df[names[:40]]
y = df["attack_type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(precision_score(y_true=y_test, y_pred=y_pred))
