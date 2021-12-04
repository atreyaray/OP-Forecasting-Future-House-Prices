import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as MAPE

def ensembling(building_type, mape_outlier_diff=0.1):
    prophet_dir = f"/u/97/nguyenq10/unix/Courses/CS-C3250 Data Science Project/OP-Forecasting-Future-House-Prices/notebooks/Ken Folder/json_prediction/{building_type}_predictions-Prophet.json"
    sarimax_dir = f"/u/97/nguyenq10/unix/Courses/CS-C3250 Data Science Project/OP-Forecasting-Future-House-Prices/notebooks/Bruce/results/{building_type}/{building_type}_sarimax_forecasts.json"

    prophet_forecast = pd.read_json(prophet_dir)
    locations = prophet_forecast[["longitude", "latitude"]]
    prophet_forecast.drop(columns=["longitude", "latitude"], inplace=True)

    sarimax_forecast = pd.read_json(sarimax_dir)
    sarimax_forecast.drop(columns=["longitude", "latitude"], inplace=True)

    df = pd.concat([sarimax_forecast, prophet_forecast], axis=1, join='inner')
    sarimax_forecast = df.iloc[:,:4]
    prophet_forecast = df.iloc[:,4:]
    err = MAPE(sarimax_forecast.T, prophet_forecast.T, multioutput='raw_values')
    outlier_bool = err > mape_outlier_diff
    
    res_df = (sarimax_forecast + prophet_forecast)/2
    res_df[outlier_bool] = prophet_forecast[outlier_bool]
    res_df = pd.concat([res_df,locations],axis=1)

    return res_df

def clustered_ensembling(building_type, mape_outlier_diff=0.1):
    prophet_dir = f"/u/97/nguyenq10/unix/Courses/CS-C3250 Data Science Project/OP-Forecasting-Future-House-Prices/notebooks/Ken Folder/json_prediction/CLUSTERED_{building_type}_predictions-Prophet.json"
    sarimax_dir = f"/u/97/nguyenq10/unix/Courses/CS-C3250 Data Science Project/OP-Forecasting-Future-House-Prices/notebooks/Bruce/results/{building_type}/{building_type}_sarimax_cluster_forecasts.json"
    
    prophet_forecast = pd.read_json(prophet_dir)
    sarimax_forecast = pd.read_json(sarimax_dir).T

    df = pd.concat([sarimax_forecast, prophet_forecast], axis=1, join='inner')
    sarimax_forecast = df.iloc[:,:4]
    prophet_forecast = df.iloc[:,4:]
    err = MAPE(sarimax_forecast.T, prophet_forecast.T, multioutput='raw_values')
    outlier_bool = err > mape_outlier_diff
    
    res_df = (sarimax_forecast + prophet_forecast)/2
    res_df[outlier_bool] = prophet_forecast[outlier_bool]

    return res_df

def clustered_forecasts():
    building_types = ["one_room", "terrace_house", "three-more_room", "two_room"]
    for building_type in building_types:
        forecasts = clustered_ensembling(building_type)
        forecasts.to_json(f'notebooks/Bruce/results/{building_type}/{building_type}_ensemble_cluster_forecasts.json')

def normal_forecasts():
    building_types = ["one_room", "terrace_house", "three-more_room", "two_room"]

    for building_type in building_types:
        forecasts = ensembling(building_type)
        forecasts.to_json(f'notebooks/Bruce/results/{building_type}/{building_type}_ensemble_forecasts.json')

def main():
    clustered_forecasts()
    normal_forecasts()

if __name__ == "__main__":
    main()



