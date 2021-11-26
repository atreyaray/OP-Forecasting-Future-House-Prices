# Basic imports
import numpy as np
import pandas as pd
import datetime # manipulating date formats
import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def sarimax_param_search(df, max_p=2, d=1, max_q=2, max_P=2, D=1, max_Q=2, s=4):
    """
    Keyword arguments:
    multivar_ts -- pandas dataframe of shape (n,m) where n is the number of
    time steps and n is the number of variables for prediction.
    """
    
    res_df = pd.DataFrame(columns=df.columns, index=list('pdqPDQs'))
    # set parameter range
    p,d,q = range(0,max_p+1),[d],range(0,max_q+1)
    P,D,Q,s = range(0,max_P+1),[D],range(0,max_Q+1),[s]

    # list of all parameter combos
    pdq = list(map(list,itertools.product(p, d, q)))
    seasonal_pdq = list(map(list, itertools.product(P, D, Q, s)))
    all_param = list(map(list, itertools.product(pdq,seasonal_pdq)))

    count = 0
    for column in df:
        best_param, best_aic = [[0,0,0],[0,0,0,4]], np.inf
        for param in all_param:
            try:
                mod = SARIMAX(
                    df[column].values,
                    order=param[0],
                    seasonal_order=param[1]
                )
                
                res = mod.fit(disp=0)
                if res.aic < best_aic:
                    best_param, best_aic = param, res.aic
            except Exception as e:
                print(e)
                continue
        
        count += 1
        res_df[column] = best_param[0] + best_param[1]

        print(f'The best model for {column} is: SARIMAX{best_param[0]}x{best_param[1]} \
            - AIC:{round(best_aic,2)}')

        print(f"Processed {round(count/len(df.columns)*100,2)}%: {count}/{len(df.columns)}")
            
    return res_df

def main():
    df = pd.read_csv("notebooks/Bruce/imputed_df.csv", index_col=0,parse_dates=[0])
    import time

    print("SARIMAX parameter search initiated...")
    start = time.time()
    sarimax_params = sarimax_param_search(df)
    end = time.time()
    print("The search took ", round(end - start,2) , "seconds")

    sarimax_params.to_csv('notebooks/Bruce/sarimax_params.csv',index=True)

if __name__ == "__main__":
    main()
