import pandas as pd
import numpy as np

def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B.xlsx", "Classification")
    # data_pd.fillna(method='ffill', inplace=True)
    data_pd.dropna(inplace=True)
    data_np = data_pd.to_numpy()
    total_size = data_np.shape[0]
    num_positive = (data_np[:,-1] == 1).sum()
    per_positive = num_positive / total_size
    print("Total size: ", total_size)
    print("Number of positive: ", num_positive)
    print("Percentage of positive: ", per_positive)
    train_size = int(total_size*0.64)
    val_size = int(total_size*0.16)
    train_data = data_np[0:train_size, :]
    val_data = data_np[train_size:train_size+val_size, :]
    test_data = data_np[train_size+val_size:, :]
    return ((train_data[:, :-1], train_data[:, -1]), 
            (val_data[:, :-1], val_data[:, -1]), 
            (test_data[:, :-1], test_data[:, -1]))
