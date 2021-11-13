import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# import data set
df = pd.read_csv('./intermediate_results/three_rooms_kmeans_rmse_mae.csv', index_col=0)
print(df)


plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Median RMSE'])
plt.xlabel('K')
plt.ylabel('RMSE')
plt.savefig('./outputs/rmse_mae_k200.png')
