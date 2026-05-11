# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:04:31 2026

@author: angel
"""

import os
import scipy.io
import numpy as np

faulty_folder = 'vibration_faults_dataset_for_rotating_machines/Faulty'

healthy_folder = 'vibration_faults_dataset_for_rotating_machines/Healthy'


def CreacionDeLista(path):
    lista = []
    for file in os.listdir(path):
        mat = scipy.io.loadmat(path+"/"+file)
        signal = mat['H']
        signal = np.array(signal)
        lista.append(signal)
        
    return lista

print(len(CreacionDeLista(faulty_folder)))
print(len(CreacionDeLista(healthy_folder)))