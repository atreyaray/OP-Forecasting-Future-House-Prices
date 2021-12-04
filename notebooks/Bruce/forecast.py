# Basic imports
import numpy as np
import pandas as pd
import datetime # manipulating date formats
import time
import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE,mean_absolute_percentage_error as MAPE

import warnings
warnings.filterwarnings("ignore")

def sarimax_forecasts(df, max_p=2, d=1, max_q=2, max_P=2, D=1, max_Q=2, s=4):
    """
    Keyword arguments:
    multivar_ts -- pandas dataframe of shape (n,m) where n is the number of
    time steps and n is the number of variables for prediction.
    """
    
    res_df = pd.DataFrame(columns=df.columns, index=['pred_0','pred_1','pred_2','pred_3'])
    # set parameter range
    p,d,q = range(0,max_p+1),[d],range(0,max_q+1)
    P,D,Q,s = range(0,max_P+1),[D],range(0,max_Q+1),[s]

    # list of all parameter combos
    pdq = list(map(list,itertools.product(p, d, q)))
    seasonal_pdq = list(map(list, itertools.product(P, D, Q, s)))
    all_param = list(map(list, itertools.product(pdq,seasonal_pdq)))

    count = 0
    for column in df:
        best_res, best_aic = None, np.inf
        for param in all_param:
            try:
                mod = SARIMAX(
                    df[column].values,
                    order=param[0],
                    seasonal_order=param[1]
                )
                
                res = mod.fit(disp=0)
                if res.aic < best_aic:
                    best_res, best_aic = res, res.aic
            except Exception as e:
                print(e)
                continue
        
        count += 1
        res_df[column] = best_res.forecast(steps=4)

        print(f'The prediction for {column} is:', res_df[column].values)

        print(f"Processed {round(count/len(df.columns)*100,2)}%: {count}/{len(df.columns)}")
            
    return res_df

def sarimax_forecast_A(building_type, prediction_type):
    df = pd.read_csv(f"notebooks/Bruce/results/{building_type}/{building_type}_imp_{prediction_type}.csv",parse_dates=[-1], index_col=[0])
    df.set_index("Quarter",inplace=True)
    df.head()

    print("SARIMAX parameter search initiated...")
    start = time.time()
    forecasts = sarimax_forecasts(df)
    end = time.time()
    print("The search took ", round(end - start,2) , "seconds")

    forecasts.to_json(f'notebooks/Bruce/results/{building_type}/{building_type}_sarimax_{prediction_type}.json', index=True)

def sarimax_forecast_B(building_type, for_validation=False):
    df = pd.read_csv(f"notebooks/Bruce/results/{building_type}/{building_type}_cluster_means.csv",parse_dates=[0], index_col=[0])
    
    if for_validation:
        df = df.iloc[:-4,:]
        output_filename = f'notebooks/Bruce/results/{building_type}/{building_type}_sarimax_cluster_forecasts_validation.json'
    else:
        output_filename = f'notebooks/Bruce/results/{building_type}/{building_type}_sarimax_cluster_forecasts.json'
    print("SARIMAX forecast initiated...")
    start = time.time()
    forecasts = sarimax_forecasts(df)
    end = time.time()
    print("The search took ", round(end - start,2) , "seconds")

    forecasts.to_json(output_filename, index=True)

def linear_regression_forecast_A(building_type, group="A"):
    df = pd.read_csv(f"notebooks/Bruce/results/{building_type}/{building_type}_imp_4_quarter_prediction.csv",parse_dates=[-1], index_col=[0])
    df.set_index("Quarter",inplace=True)
    dfT = df.T
    X,y = dfT.iloc[:,:-4], dfT.iloc[:,-4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mae = MAE(y_test.T, y_pred.T, multioutput='raw_values')
    mape = MAPE(y_test.T, y_pred.T, multioutput='raw_values')

    print(building_type)
    print("MAE avg: ", round(np.mean(mae),3), "MAPE avg: ", round(np.mean(mape),3))
    print("MAE median: ", round(np.median(mae),3), "MAPE median: ", round(np.median(mape),3))

if __name__ == "__main__":
    # forecast_group_A("two_room", "train")
    building_types = ["one_room", "terrace_house", "three-more_room", "two_room"]

    for building_type in building_types:
        linear_regression_forecast_A(building_type)

    
