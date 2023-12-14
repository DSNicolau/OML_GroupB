import sys

sys.path.append("Python/classification/")

import utils
import preprocessing

if __name__ == "__main__":
    train, val, test = utils.load_data(fill_method="ffill")
    x = preprocessing.mutual_information(train[0][:, 5:], train[1])
    weeks = preprocessing.get_week_day(train[0])
    df_heatmap = preprocessing.number_events(x=weeks,x_name=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] ,y=train[0][:,3],events=train[1])
    print('ah')
