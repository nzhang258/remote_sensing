import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import time

def norm(dt, cen):
  tmp = 0
  for n in range(dt.shape[0]):
    tmp += np.linalg.norm(dt[n,:]-cen)
  tmp /= dt.shape[0]
  return tmp

colors = ['r*', 'g*', 'b*']
def plot(epoch):
  fp = open('../data/t0/epoch_{}.pkl'.format(epoch),'rb')
  data = pkl.load(fp)
  fp.close()
  centroids = np.array(data['centroids'])
  print('centrodis', centroids)
  for i in range(2):
    dt = np.array(data['data'][i])
    if dt.shape[0] > 0:
      plt.plot(dt[:,0], dt[:,1],colors[i])  
      print('norm', norm(dt, data['centroids'][i]))
  for i in range(2):
    plt.plot(centroids[i,0], centroids[i,1], 'ko', markersize=24 )
    plt.plot(centroids[i,0], centroids[i,1], colors[i], markersize=20 )
  plt.show()
# plt.ion()

for epoch in range(300):
  plot(epoch)
  # time.sleep(2)