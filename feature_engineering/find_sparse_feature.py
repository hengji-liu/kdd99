from data.data import Data


class SparseFeatureFinder:
    data = Data("full")
    df = data.df
    services = data.services
    flags = data.flags
    protocol_types = data.protocol_types

    @classmethod
    def check_protocol_type(cls):
        '''
        find if there is a 1 to 1 mapping between protocol type and attack type
        :return:
        '''
        print(cls.protocol_types)
        for pt in cls.protocol_types:
            attack_type_list = cls.df[(cls.df["protocol_type"] == pt)].attack_type.unique()
            if len(attack_type_list) == 1:
                print(pt, attack_type_list[0])
                # RESULT:
                # nothing

    @classmethod
    def check_service(cls):
        print(cls.services)
        for servo in cls.services:
            attack_type_list = cls.df[(cls.df["service"] == servo)].attack_type.unique()
            if len(attack_type_list) == 1:
                print(servo, attack_type_list[0])
                # RESULT:
                # ntp_u     normal.
                # pm_dump   satan.
                # urh_i     normal.
                # http_2784 satan.
                # harvest   satan.
                # aol       satan.
                # tftp_u    normal.
                # http_8001 satan.
                # red_i     normal.
                '''
                could merge into 2 groups to reduce feature dimension after one-hot encoding
                '''

    @classmethod
    def check_flag(cls):
        print(cls.flags)
        for flag in cls.flags:
            attack_type_list = cls.df[(cls.df["flag"] == flag)].attack_type.unique()
            if len(attack_type_list) == 1:
                print(flag, attack_type_list[0])
                # RESULT:
                # RSTOS0 portsweep.


if __name__ == '__main__':
    # SparseFeatureFinder.check_protocol_type()
    SparseFeatureFinder.check_service()
    # SparseFeatureFinder.check_flag()
