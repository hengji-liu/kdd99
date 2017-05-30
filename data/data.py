import numpy as np
import pandas as pd


class Data:
    __FILE10PC = r"../data/train10pc"
    __FILE = r"../data/train"
    __ATTR_NAMES = ("duration",  # length (number of seconds) of the conn's
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
                    )

    # protocol_type = ['tcp' 'udp' 'icmp']
    #
    # services = ['http' 'smtp' 'domain_u' 'auth' 'finger' 'telnet' 'eco_i' 'ftp' 'ntp_u'
    # 'ecr_i' 'other' 'urp_i' 'private' 'pop_3' 'ftp_data' 'netstat' 'daytime'
    # 'ssh' 'echo' 'time' 'name' 'whois' 'domain' 'mtp' 'gopher' 'remote_job'
    # 'rje' 'ctf' 'supdup' 'link' 'systat' 'discard' 'X11' 'shell' 'login'
    # 'imap4' 'nntp' 'uucp' 'pm_dump' 'IRC' 'Z39_50' 'netbios_dgm' 'ldap'
    # 'sunrpc' 'courier' 'exec' 'bgp' 'csnet_ns' 'http_443' 'klogin' 'printer'
    # 'netbios_ssn' 'pop_2' 'nnsp' 'efs' 'hostnames' 'uucp_path' 'sql_net'
    # 'vmnet' 'iso_tsap' 'netbios_ns' 'kshell' 'urh_i' 'http_2784' 'harvest'
    # 'aol' 'tftp_u' 'http_8001' 'tim_i' 'red_i']
    #
    # flags = ['SF' 'S2' 'S1' 'S3' 'OTH' 'REJ' 'RSTO' 'S0' 'RSTR' 'RSTOS0' 'SH']
    #
    # attack_types = ['normal.' 'buffer_overflow.' 'loadmodule.' 'perl.' 'neptune.' 'smurf.'
    # 'guess_passwd.' 'pod.' 'teardrop.' 'portsweep.' 'ipsweep.' 'land.'
    # 'ftp_write.' 'back.' 'imap.' 'satan.' 'phf.' 'nmap.' 'multihop.'
    # 'warezmaster.' 'warezclient.' 'spy.' 'rootkit.']

    def __init__(self, filename):
        if filename == 'full':
            self.df = pd.read_csv(self.__FILE, header=None, names=self.__ATTR_NAMES)
        elif filename == "10pc":
            self.df = pd.read_csv(self.__FILE10PC, header=None, names=self.__ATTR_NAMES)
        else:
            raise Exception("specify 10pc or full")

        self.services = self.df.service.unique()
        self.flags = self.df.flag.unique()
        self.protocol_types = self.df.protocol_type.unique()
        self.attack_types = self.df.attack_type.unique()

    def size(self):
        return self.df.size

    def shape(self):
        return self.df.shape

    # https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    def describe(self):
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print("---")
        print(self.df.describe(include=[np.object_]))  # categorical features
        print("---")
        print(self.df.describe(include=[np.number]))  # numeric features
        print("---")
