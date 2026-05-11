# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:19:20 2026

@author: angel
"""
# Librerias
import os
import scipy.io
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

# Declaro las direcciones de los archivos .mat de las señales de motores
# Los que fallan (faulty) y los que no (healthy), son carpetas separadas
faulty_folder = 'vibration_faults_dataset_for_rotating_machines/Faulty'
healthy_folder = 'vibration_faults_dataset_for_rotating_machines/Healthy'

# Creo una funcion para generar dataframes
# A partir de los archivos de señales
# Tambien como argumento se le agrega si esta dañado o no (failt)
def CreacinDataFrame(path, fault):
    ''' 
    Se lee archivo por archivo (.mat) de una carpete
    se obtiene 3 señales crudas de un acelerometro(sensor de "vibracion"): 
        "x", "y" y "z"
    Tambien se cran 4 señales mas a partir de estas:
        sum_signal = suma de las 3 señales crudas(x,y,z)
        mean_signal = media de las 3 señales
        s_signal = señal sigma funcion obtenida de su documentacion (norma euclidiana)
        s_signal_normalized = la funcion anterior pero normalizada (0 - 1)
    Al final agregue una 8a columna que declara si tiene falla o no declarado en el argumento
    Cree una lista con de mini diccionarios con esta informacion
    Al final se junto todo en un solo dataframe que es lo que retorna.
    '''
    df_list = []
    for file in os.listdir(path):
        mat = scipy.io.loadmat(path+"/"+file)
        signal = np.array(mat['H'])
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
                'fault':fault
                }
        
        df_list.append(data)

    return pd.DataFrame(df_list)

# Creo el dataframe para los motores sin fallas
df_healthy = CreacinDataFrame(healthy_folder,0)
# Creo el dataframe para los motores con fallas
df_faulty = CreacinDataFrame(faulty_folder,1)

# Combino ambos df
df_combinado = pd.concat([df_healthy, df_faulty], axis=0)
# Los revuelvo
df = df_combinado.sample(frac=1, random_state=45).reset_index(drop=True)

print(df.shape)
print(df.head())
#plt.plot(range(len(df_healty['s_signal'][0])),df_healty['s_signal'][0])
#plt.show()
