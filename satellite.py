#!/usr/bin/env python

import numpy as np 
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
import pandas as pd
import colorsys
import pickle

svc = pickle.load(open('model.dat', 'rb'))
data = loadmat("/classes/ece2720/fpp/test_x_only.mat")
test_x = data["test_x"]
test_x_shaped = test_x.reshape(3136, 100000).T

running_len = 10000

x_hsv = np.empty((28, 28, 4, running_len))
for i in range(running_len):
    for row in range(28):
        for col in range(28):
            r = int(test_x[row, col, 0, i]) / 255.
            g = int(test_x[row, col, 1, i]) / 255. 
            b = int(test_x[row, col, 2, i]) / 255.
            nir = int(test_x[row, col, 3, i])
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            x_hsv[row, col, 0, i] = h 
            x_hsv[row, col, 1, i] = s 
            x_hsv[row, col, 2, i] = v 
            x_hsv[row, col, 3, i] = nir 

x_hsv_avg = np.empty((8, running_len))
for i in range(running_len):
    values_lst = [] 
    values_lst.append(np.mean(x_hsv[:, :, 0, i]))
    values_lst.append(np.mean(x_hsv[:, :, 1, i]))
    values_lst.append(np.mean(x_hsv[:, :, 2, i]))
    values_lst.append(np.mean(x_hsv[:, :, 3, i]))
    values_lst.append(np.std(x_hsv[:, :, 0, i]))
    values_lst.append(np.std(x_hsv[:, :, 1, i]))
    values_lst.append(np.std(x_hsv[:, :, 2, i]))
    values_lst.append(np.std(x_hsv[:, :, 3, i]))
    for j in range(8):
        x_hsv_avg[j, i] = values_lst[j]  

y_pred = svc.predict(x_hsv_avg.T)

L = ['barren land', 'trees', 'grassland', 'none']
s = ','.join([L[t] for t in y_pred])
f = open('landuse.csv', 'w')
f.write(s)
f.close()





