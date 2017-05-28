import pandas as pd


def transform(df):
    service_one_hot = pd.get_dummies(df["service"])
    df = df.drop('service', axis=1)
    df = df.join(service_one_hot)

    protocol_type_one_hot = pd.get_dummies(df["protocol_type"])
    df = df.drop('protocol_type', axis=1)
    df = df.join(protocol_type_one_hot)

    flag_type_one_hot = pd.get_dummies(df["flag"])
    df = df.drop('flag', axis=1)
    df = df.join(flag_type_one_hot)
    return df
