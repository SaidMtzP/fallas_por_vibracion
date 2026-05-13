# Diagnostico de fallas en motores

## Objetivo
En este proyecto se busca diagnosticar la falla de motores por medio de vibraciones.
Toda la informacion acerca de este proyecto y el dataset se encuentran en los links de acontinuacion:
  
  Dataset: https://www.kaggle.com/datasets/sumairaziz/vibration-faults-dataset-for-rotating-machines
  
  Documentacion original: https://doi.org/10.3390/s21227587

## Descripcion del dataset


El dataset con el que trabajé es un conjunto de 320 archivos de matlab(.mat), 103 de motores saludables y 117 con falla, cada uno de estos contiene 3 señales de vibracion en el eje x, y, z; que fueron adquiridos por medio de un acelerometro. Cada señal son vibraciones por eje, es decir posicionamiento o diferencia de posicionamiento en x, y, z; mientras el motor esta encendido, cada muetra esta tomada a una frecuencia de 1000Hz durante 5 segundos y cada dato guardado esta registrado por cada milisegundo. En total, por archivo, tengo un array de 3 señales con 5000 datos cada una.

## Desarrollo del proyecto


Como el dataset adquirido son un conjunto de archivos .mat, de los cuales unos eran de motores con fallas y los otros de motores saludables, primero tuve que lograr obtener unicamente las señales(x,y,z), y de forma repetitiva, extraer cada una de estas por cada archivo. Primero obtuve la señal sigma que es la normal euclidiana o la raiz cuadrada de la suma de los cuadrados de sus componentes, esta señal fue calculada por recomendacion del dueño del dataset. 

Como serían muchos datos trabajar con un dataframe lleno de 3 señales por archivo con muestras de 5000 datos (dato por milisegndo), decidí procesar cada señal para obtener el valor minimo, máximo, RMS, el valor de Curtosis y el dominio de la frecuencia por cada señal, logrando tener solo 5 datos por cada señal, y son 4 señales (x,y,z) obtuve 20 datos por cada archivo, al final se agrego un dato mas que indica si el motor tiene falla o no.
Hice esto por cada archivo de motor con fallo y sin fallo. Al final se trabajo con un dataset de 220 x 21.

A continuación muestro una imagen del comportamiento de los motores con fallas y sin fallas:
<img width="780" height="564" alt="image" src="https://github.com/user-attachments/assets/024a0834-b523-4ae4-bc69-879a4ca6fb3c" />
 
## Resultado

Implemente un modelo de potenciacion de gradiante con la librería XGBoost, el cual se califico con la validación cruzada y el valor F1 que indica su precision y esactitud. Obtuve los siguientes resultados:

<img width="550" height="50" alt="image" src="https://github.com/user-attachments/assets/2d8cd0ab-12cb-4172-b613-e7e70311bdb3" />


## Conclusion

Con esto a pesar de que es diferente a lo que se menciona en el paper, ya que toma otro camino diferente; puedo decir que el modelo de XGBoost da buenos resultados con alta validación y buen nivel de precision y exactitud, dejando claro que puede predecir con mucha fiabilidad si un motor tiene fallas en futuras ocaciones.
