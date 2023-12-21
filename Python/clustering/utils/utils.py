import pandas as pd


def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B_v2.xlsx", "Clustering")
    data_pd.dropna(inplace=True)
    return data_pd

