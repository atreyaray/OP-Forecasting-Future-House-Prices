import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# import data set
df = pd.read_csv('./intermediate_results/three_rooms_kmeans_rmse_mae.csv', index_col=0)
print(df)


plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Median RMSE'])
plt.xlabel('K')
plt.ylabel('Median RMSE')
plt.title('Median RMSE by Cluster Size (k)')
plt.savefig('./outputs/rmse_k200.png')


plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Median MAE'])
plt.xlabel('K')
plt.ylabel('Median MAE')
plt.title('Median MAW by Cluster Size (k)')
plt.savefig('./outputs/mae_k200.png')
