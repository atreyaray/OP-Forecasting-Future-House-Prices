#import libraries required

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from tslearn.metrics import dtw
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import re
import statistics



# get clustered data
clustered_df = pd.read_csv('./intermediate_results/three_rooms_kmeans_clustered.csv', index_col=0)
dates = pd.read_csv('./intermediate_results/ds.csv', index_col=0)

col = clustered_df.columns.values
col = np.concatenate((dates.ds.values, col[46:]), axis=0)
clustered_df.columns = col

# get data
data = clustered_df.iloc[:, :46].copy()

K = range(2,3)
m_df = pd.DataFrame(data = [], columns = ["K", "Median RMSE", "Median MAE"])


def prophet_for_each_postalcodes(df, postal_codes, time="ds"):

    # error values
    mae_values = []
    rmse_values = []

    for pc in postal_codes:
        print(pc)
        # rename columns for prophet         
        pc_df = df[[pc, time]].rename(columns={pc:"y",time:"ds"})
        
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

    return mae_values, rmse_values

# postal_codes = ts_df.columns[:-1]
# data, mae_values, rmse_values = prophet_for_each_postalcodes(ts_df, postal_codes)

for k in K:
    print('K=', k)
    # add column of labels
    data['K'] = clustered_df["K="+str(k)]

    # groupby cluster
    k_df = data.groupby('K').mean().T
    
    k_df['ds'] = dates.ds.values
    postal_codes = list( range(k))

    mae_values, rmse_values = prophet_for_each_postalcodes(k_df, postal_codes)

    median_rmse = statistics.median(rmse_values)
    median_mae = statistics.median(mae_values)
    m_df = m_df.append({"K":k, "Median RMSE": median_rmse, "Median MAE": median_mae}, ignore_index = True)

m_df.to_csv('./intermediate_results/three_rooms_kmeans_rmse_mae.csv')
