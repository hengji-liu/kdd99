import numpy as np

CLASS_1 = ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.']
CLASS_2 = ['apache2.', 'back.', 'mailbomb.', 'neptune.', 'pod.', 'land.'
           'processtable.', 'smurf.', 'teardrop.', 'udpstorm.', 'mailbomb.']
CLASS_3 = ['buffer_overflow.', 'loadmodule.',
           'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.']
CLASS_4 = ['ftp_write.', 'guess_passwd.', 'httptunnel.', 'imap.', 'multihop.',
           'named.', 'phf.', 'sendmail.', 'snmpgetattack.', 'snmpguess.',
           'worm.', 'xlock.', 'xsnoop.', 'spy.', 'warezclient.', 'warezmaster.']

y = np.ndarray


def map_to_major_classes(y):
    classes = list()
    for cl in y:
        if cl in CLASS_1:
            classes.append(1)
        elif cl in CLASS_2:
            classes.append(2)
        elif cl in CLASS_4:
            classes.append(4)
        elif cl in CLASS_3:
            classes.append(3)
        else:
            classes.append(0)

    return classes






