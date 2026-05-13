# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:19:20 2026

@author: angel
"""
# Librerias
import os
import scipy.io
from scipy.stats import kurtosis
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Librerias de ml
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score



# Declaro las direcciones de los archivos .mat de las señales de motores
# Los que fallan (faulty) y los que no (healthy), son carpetas separadas
faulty_folder = 'vibration_faults_dataset_for_rotating_machines/Faulty'
healthy_folder = 'vibration_faults_dataset_for_rotating_machines/Healthy'

# Creo una funcion para generar la funcion sigma
# Que en realidad es Norma Euclidiana, la raiz cuadrada de la suma 
# de los cuadrados de sus componentes
# Se ingresa la señal con 3 vectores (x, y, z) y se regresa la señal sigma
def sigma_function(signal):
    s_signal = []
    s_signal_normalized = []
    for i in range(signal.shape[0]):
        s_signal.append(signal[i].dot(signal[i].T).item()**0.5) # Funcion sigma
        #s_signal_normalized.append(s_signal[i]/max(s_signal))
        # Se centra la señal para poderse utilizar despues
        s_signal_normalized.append(s_signal[i] - np.mean(s_signal))
        
    return s_signal_normalized

# Funcion para obtener la frecuencia 
# Se introduce la señal y la frecuencia de muestra que en este caso es 1000Hz
def obtener_frecuencia_dominante(signal, fs):
    n = len(signal)
    # Calculamos la FFT y las frecuencias asociadas
    fft_data = np.fft.rfft(signal) 
    frecuencias = np.fft.rfftfreq(n, d=1/fs)
    
    # Obtenemos la magnitud y buscamos el índice del pico máximo
    magnitud = np.abs(fft_data)
    indice_max = np.argmax(magnitud)
    
    return frecuencias[indice_max]

# Funcion para obtener cada una de las caracteristicas especiales de una señal
# El RMS, el valor Kurtosis, y el dominio de la frecuencia
# Se itroduce la señal y se obtienen los 3 datos anteriores
def GetFeatures(signal):
    signal = np.array(signal) # La señal se combierte en array
    rms = np.sqrt(np.mean(signal**2)) # Se obtiene el RMS
    v_kurtosis = kurtosis(signal, fisher=False) # Se obtiene el valor Kurtosis
    f_dom = obtener_frecuencia_dominante(signal, fs=1000) # Dominio de Frec.
    
    return rms, v_kurtosis, f_dom # Se regresan los 3 valores al mismo tiempo

# Creo una funcion para generar dataframes
# A partir de los archivos de señales
# Tambien como argumento se le agrega si esta dañado o no (fault)
def CreacinDataFrame(path, fault):
    ''' 
    Se lee archivo por archivo (.mat) de una carpete
    se obtiene 3 señales crudas de un acelerometro(sensor de "vibracion"): 
        "x", "y" y "z"
    Se obtiene la señal sigma con estas 3 señales.
    De estas 4 se cran 5 caracteristicas por cada una:
        max = Valor max de la señal
        min = Valor minimo de la señal
        RMS = RMS de la señal
        Kurtosis = Valor Kurtosis (Forma pico de la señal)
        F_dom = Dominio de la frecuencia de la señal
    Al final agregue una 21a columna que declara si tiene falla o 
        no declarado en el argumento
    Cree una lista de diccionarios con esta informacion
    Al final se junto todo en un solo dataframe que es lo que retorna.
    '''
    df_list = [] # Lista donde se guardaran los diccionarios de datos
    for file in os.listdir(path):
        mat = scipy.io.loadmat(path+"/"+file) # Se abre el archivo .mat
        signal = np.array(mat['H']) # Se extrae unicamente la señal

        s_signal = sigma_function(signal) # Se crea la señal sigma
        
        # Se crea la variable 'names' para crear variables diferentes
        # con prefijos x,y,z
        names = ['x','y','z'] 
        data = {} # Se inicia el diccionario
        for p in names: # Se crea un bucle para obtener los datos de cada señal
            for i in range(signal.shape[1]):
                data[f"{p}_max"] = np.max(signal[:,i]) # max
                data[f"{p}_min"] = np.min(signal[:,i]) # min
                # RMS, Kurtosis, f_dom
                data[f"{p}_rms"], data[f"{p}_kurt"], data[f"{p}_f_dom"] = GetFeatures(signal[:,i])
            
        # Por ultimo se extrae lo mismo pero de la señal sigma
        data["s_max"] = np.max(s_signal)
        data["s_min"] = np.min(s_signal)
        data["s_rms"], data["s_kurt"], data["s_f_dom"] = GetFeatures(s_signal)
        
        # Se llena el ultimo campo indicando si es motor con falla o no
        data["fault"] = fault
        
        df_list.append(data) # Se guarda los diccionarios en la lista

    return pd.DataFrame(df_list) # Se crea el dataframe y se retorna

# Creo el dataframe para los motores sin fallas
df_healthy = CreacinDataFrame(healthy_folder,0)
# Creo el dataframe para los motores con fallas
df_faulty = CreacinDataFrame(faulty_folder,1)

# Combino ambos df
df_combinado = pd.concat([df_healthy, df_faulty], axis=0)
# Los revuelvo
df = df_combinado.sample(frac=1, random_state=45).reset_index(drop=True)

#print(df.shape)
#print(df.head())
X = range(df_healthy.shape[0])
X1 = range(df_faulty.shape[0])
plt.plot(X,df_healthy['s_rms'])
plt.plot(X1,df_faulty['s_rms'])
plt.title('Comparación de comportamiento')
plt.xlabel('Señales')
plt.ylabel('RMS')
plt.legend(["Saludable",'Con falla'])
plt.show()

df_features = df.drop('fault',axis=1) # Se genera el df de las caracteristicas
df_target = df['fault'] # Y el del objetivo


# Se separa los df en entrenamiento y test
features_train, features_test, target_train, target_test = train_test_split(
    df_features, df_target, test_size=0.2,random_state=45
    )

# Se crea el modelo de clasificacion por potenciacion de gradiante usando XGBoost
modelo_gb = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5,
    #use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

# Se entrena el modelo
modelo_gb.fit(features_train,target_train)
# Se predice com el df de prueba
predictions_valid = modelo_gb.predict(features_test)

# Se califica el modelo usando la validacion cruzada
final_score = (cross_val_score(modelo_gb, features_train,target_train).sum())/5

# Se imprime el valor de la validacion cruzada y su valor F1(Precision y exactitud)
print(f'Puntuación media de la evaluación del modelo:{final_score:.2f}')
print(f'Valor de F1:{f1_score(target_test,predictions_valid):.2f} ')
