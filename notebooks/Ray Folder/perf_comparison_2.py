from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from prophet import Prophet
from prophet.plot import plot_plotly

# read csv file
df = pd.read_csv("../Ken Folder/df_geo_imp.csv", index_col=0)
# rename column names
df.columns = ["PC", "BT", "Quarter", "EUR/m2", "latitude", "longitude"]

# create dataframes of different room types
one_room = df[df["BT"] == "one-room"]
two_room = df[df["BT"] == "two-room"]
three_room = df[df["BT"] == "three or more room"]

# Use 3-room apartments, create time series
ts_df = pd.DataFrame()
grouping = three_room.groupby('PC')
for i in df.PC.unique():
    ts_df[str(i)] = grouping.get_group(i)["EUR/m2"].values

ds = pd.read_csv('./intermediate_results/ds.csv', index_col=0)
# print("ds")
# print(ds)

ts_df['ds'] = ds['ds'].values
print("ts_df")
print(ts_df)


def prophet_for_each_postalcodes(df, postal_codes, time="ds"):

    # error values
    mae_values = []
    rmse_values = []

    for pc in postal_codes:
        print(pc)
        # rename columns for prophet
        pc_df = df[[pc, time]].rename(columns={pc: "y", time: "ds"})

        # divide dataset into train and test
        train, test = pc_df[:-4], pc_df[-4:]
        # print(train)
        # print(test)

        # Initialise model, train and predict
        prop = Prophet(interval_width=0.95)
        prop.fit(train)
        forecast = prop.predict(test)

        # compute errors
        mae = mean_absolute_error(test["y"], forecast["yhat"])
        rmse = math.sqrt(mean_squared_error(test["y"], forecast["yhat"]))

        # update error lists
        mae_values.append(mae)
        rmse_values.append(rmse)

    data = pd.DataFrame({"postal code": postal_codes,
                         "rmse": rmse_values, "mae": mae_values})
    return data, mae_values, rmse_values


postal_codes = ts_df.columns[:-1]
data, mae_values, rmse_values = prophet_for_each_postalcodes(ts_df, postal_codes[:4])
data.to_csv('./intermediate_results/three_rooms_pc_rmse_mae.csv', index=False)
print('final output')
print(data)
