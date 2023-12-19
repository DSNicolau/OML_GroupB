import pandas as pd


def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B.xlsx", "Clustering").dropna(inplace=True)
    return data_pd

