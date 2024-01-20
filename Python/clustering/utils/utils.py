import pandas as pd


def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B_v2.xlsx", "Clustering")
    data_pd.dropna(inplace=True)
    return data_pd


def min_max_nomalization(data):
    min_features = data.min(axis=0)
    max_features = data.max(axis=0)
    data = (data - min_features) / (max_features - min_features)
    return data


def min_max_nomalization_pandas(data):
    data_norm = pd.DataFrame()
    for key in data.keys():
        data_norm[key] = (data[key] - data[key].min()) / (
            data[key].max() - data[key].min()
        )
    return data_norm
