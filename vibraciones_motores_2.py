# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:19:20 2026

@author: angel
"""

import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

mat = scipy.io.loadmat(r'C:\Users\angel\OneDrive\Documents\datasets\vibration_faults_dataset_for_rotating_machines\Faulty/F1.mat')

signal = mat['H']

signal = np.array(signal)

suma_signal = np.sum(signal, axis=1).tolist()
mean_signal = np.mean(signal,axis=1).tolist()

s_signal = []
s_signal_normalized = []
for i in range(signal.shape[0]):
    s_signal.append(signal[i].dot(signal[i].T).item()**0.5)    
    s_signal_normalized.append(s_signal[i]/max(s_signal))

data = {'x':signal[:,0].tolist(),
        'y':signal[:,1].tolist(),
        'z':signal[:,2].tolist(),
        'sum_signal':suma_signal,
        'mean_signal':mean_signal,
        's_signal':s_signal,
        's_signal_normalized':s_signal_normalized,
        'fault':1
        }

df_list = [data]

df = pd.DataFrame(df_list)

#print(df['s_signal'][0][2])

#
plt.plot(range(len(df['s_signal'][0])),df['s_signal'][0])
plt.show()