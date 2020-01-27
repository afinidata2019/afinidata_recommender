import numpy as np


def shuffle_columns(df):
    for column in df.columns:
        df[column] = np.random.permutation(df[column].values)
    return df
