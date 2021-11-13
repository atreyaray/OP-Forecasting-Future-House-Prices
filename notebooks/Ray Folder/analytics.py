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
plt.savefig('./outputs/clustering/rmse_median_k200.png')


plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Median MAE'])
plt.xlabel('K')
plt.ylabel('Median MAE')
plt.title('Median MAE by Cluster Size (k)')
plt.savefig('./outputs/clustering/mae_median_k200.png')


K = range(2,200)
rmse_list = []
mae_list = []

for k in K :
  mean_rmse_k = []
  mean_mae_k = []
  df =  pd.read_csv('./intermediate_results/errors/k=' + str(k)+'.csv', index_col = 0)
  for i in df.index :
    mean_rmse_k.append(df.loc[i, 'RMSE'])
    mean_mae_k.append(df.loc[i, 'MAE'])

  mean_rmse = np.mean(np.array(mean_rmse_k))
  mean_mae = np.mean(np.array(mean_mae_k))
  
  rmse_list.append(mean_rmse)
  mae_list.append(mean_mae)


df = pd.DataFrame({'K': list(K), 'Mean RMSE': rmse_list, 'Mean MAE': mae_list})

plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Mean RMSE'])
plt.xlabel('K')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE by Cluster Size (k)')
plt.savefig('./outputs/clustering/rmse_mean_k200.png')


plt.figure(figsize = (10,4))
plt.plot(df['K'], df['Mean MAE'])
plt.xlabel('K')
plt.ylabel('Mean MAE')
plt.title('Mean MAE by Cluster Size (k)')
plt.savefig('./outputs/clustering/mae_mean_k200.png')

print(df)

