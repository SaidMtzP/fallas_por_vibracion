# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def Sdt(row):
    S = (((row.iloc[0])**2)+((row.iloc[1])**2)+((row.iloc[2])**2))**0.5
    return S


mat = scipy.io.loadmat(r'C:\Users\angel\OneDrive\Documents\datasets\vibration_faults_dataset_for_rotating_machines\Faulty/F1.mat')

senal = np.array(mat['H'])

#senal = senal.flatten() 
print(senal[0])

df = pd.DataFrame(senal,columns=('x','y','z'))

df['suma'] = df[['x','y','z']].sum(axis=1)
df['mean'] = df[['x','y','z']].mean(axis=1)
df['S'] = df.apply(Sdt,axis=1)
df['s_normalized'] = df['S']/df['S'].max()


print(df.head())


plt.figure(num=1,figsize=(20,10))
plt.plot(df['mean'])
plt.show()
