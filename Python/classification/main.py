import pandas as pd



if __name__ == "__main__":
    data = pd.read_excel("data/Datasets_Group_B.xlsx", "Classification")
    train_data = data.iloc[:int(len(data)*0.64)]
    vali_data = data.iloc[int(len(data)*0.64):int(len(data)*0.8)]
    test_data = data.iloc[int(len(data)*0.8):]
    