#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Visión Computacional </em><br>
#     <em>Profesor: Takeshi Asahi </em><br>
# 
# </div>
# 
# # Machine Learning Avanzado
# *22 de Abril de 2024*
# 
# #### Integrantes: 
# ` Giuseppe Lavarello`

# ## 1. Objetivo
# 
# El objetivo principal de esta tarea consiste en implementar una serie de pasos que concluyan en la creación de un modelo de clasificación para el conjunto de datos **Brain Tumor MRI Dataset**, disponible en **Kaggle**.
# 
# 1. Se eligió este dataset por su relevancia en el tratamiento de pacientes con tumores cerebrales. Este es un problema importante, pues la identificación de los tumores facilita la determinación del tratamiento con mayores probabilidades de éxito.
# 2. El propósito es identificar los tipos de tumores presentes en las imágenes obtenidas mediante resonancia magnética, entre: tumores pituitarios, gliomas, meningiomas o la ausencia de tumor.
# 3. Pasos a seguir:
#     - Diseñar un flujo de procesamiento: adquisición de imágenes, preprocesamiento, procesamiento de imágenes, cálculo de mediciones y almacenamiento.
#     - Implementar un sistema de clasificación mediante CNN.
#     - Medir los tiempos de ejecución del procesamiento de las imágenes de prueba.
#     - Hacer un resumen de los resultados del procesamiento, en particular de sus métricas.
# 

# # 2. Carga de datos

# manejo datos
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# visualización y gráficos
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# Os y glob para manejo de archivos
import os
from glob import glob

# Modelos
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D

# Configuración de TensorFlow para evitar advertencias y errores de GPU
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import absl.logging
absl.logging.set_verbosity('error')  # or 'fatal'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#medicion de ejecucion
import time


# **Decisión**  
# En esta tarea, a diferencia de la anterior, se creará un dataframe que contenga el `path` y la `clase` de cada imagen. Esto permitirá realizar el preprocesamiento directamente con la librería `ImageDataGenerator` de Keras.
# 

def load_paths(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes}) # DataFrame con rutas y etiquetas
    return tr_df


tr_df = load_paths(r'./data/Training')
ts_df = load_paths(r'./data/Testing')


# Visualización de las clases y sus conteos
def plot_class_distribution(df):
    """ Visualiza la distribución de clases en el conjunto de entrenamiento.
    Args:
        tr_df (DataFrame): DataFrame que contiene las rutas de las imágenes y sus etiquetas.
    """

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))

    ax = sns.countplot(data=df, x='Class', order=tr_df['Class'].value_counts().index)

    ax.set_title('Distribución de Clases en el Conjunto de Entrenamiento')
    ax.set_xlabel('Clases')
    ax.set_ylabel('Conteo')
    plt.xticks(rotation=25)


    # añadir cantidad dentro de las barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()//2), 
                    ha='center', 
                    va='bottom', 
                    fontsize=15, 
                    color='w', 
                    weight='bold', 
                    rotation=90, 
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    #añadir total de imágenes arriba a la derecha 
    total = len(df)
    ax.annotate(f'Total de imagenes: {total}', 
                (0.9, 0.90), 
                xycoords='axes fraction', 
                ha='right', 
                va='top', 
                fontsize=15, 
                color='black',  
                rotation=0, 
                xytext=(0, 5), 
                textcoords='offset points')
    plt.tight_layout()
    plt.show()


# **Decisión**  
# Dado que no hay datos de validación, separamos aleatoriamente los datos de prueba en dos grupos: 50% para validación y 50% para prueba.
# 

val_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=1122, stratify=ts_df['Class'])


# En los siguientes diagramas podemos ver que el dataset se compone de 3 grupos: `ts_df`, `val_df` y `tr_df`. Cada uno contiene imágenes de las 4 clases presentes en los datos: `notumor`, `pituitary`, `meningioma` y `glioma`. El conjunto `tr_df` consiste en 5712 imágenes, `ts_df` en 656 imágenes y `val_df` en una cantidad de imágenes no especificada.
# 

plot_class_distribution(tr_df)


plot_class_distribution(val_df)


plot_class_distribution(ts_df)


# **Importante**  
# Aunque existe un pequeño desbalance en los datos a favor de la clase `notumor`, este no es lo suficientemente significativo como para justificar una limpieza más profunda en esta etapa. Sin embargo, en las siguientes iteraciones de esta tarea se trabajará para tratar de obtener mejores resultados.
# 

tr_df.describe()


val_df.describe()


ts_df.describe()


# # 3. Preprocesamiento
# Se crean los manipuladores de imagen y se realizan tres tratamientos distintos para generar diferentes conjuntos de datos. 
# 
# 1. En la primera iteración, las imágenes son reescaladas y convertidas a escala de grises.
# 
# 

batch_size = 32
img_size = (256, 256)

_gen = ImageDataGenerator(rescale=1/255)

ts_gen = ImageDataGenerator(rescale=1/255)


tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                    y_col='Class', batch_size=batch_size,
                                    target_size=img_size, # reescalado de imagenes
                                    color_mode='grayscale', # cambio a greyscale

                                  )

valid_gen = _gen.flow_from_dataframe(val_df, x_col='Class Path',
                                        y_col='Class', batch_size=batch_size,
                                        target_size=img_size, # reescalado de imagenes)
                                        color_mode='grayscale', # cambio a greyscale

                                     )

ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                    y_col='Class', batch_size=16,
                                    target_size=img_size, # reescalado de imagenes)
                                    color_mode='grayscale', # cambio a greyscale
                                    shuffle=False)




# 2. En la segunda iteración, las imágenes son rotadas aleatoriamente entre 0 y 90 grados.
# 

_gen_rotated = ImageDataGenerator(rescale=1/255, rotation_range=90)

tr_gen_rotated = _gen_rotated.flow_from_dataframe(tr_df, x_col='Class Path',
                                    y_col='Class', batch_size=batch_size,
                                    target_size=img_size, # reescalado de imagenes
                                    color_mode='grayscale', # cambio a greyscale
                                    class_mode='categorical', # clasificacion multiclase
                                  )

valid_gen_rotated = _gen_rotated.flow_from_dataframe(val_df, x_col='Class Path',
                                        y_col='Class', batch_size=batch_size,
                                        target_size=img_size, # reescalado de imagenes)
                                        color_mode='grayscale', # cambio a greyscale
                                        class_mode='categorical', # clasificacion multiclase
                                     )


# 3. En la tercera iteración, las imágenes son rotadas aleatoriamente entre 0 y 90 grados, y trasladadas también de forma aleatoria hasta un 20% tanto en alto como en ancho.
# 

_gen_rotated_shifted = ImageDataGenerator(rescale=1/255, rotation_range=90,width_shift_range=0.2, height_shift_range=0.2)

tr_gen_rotated_shifted = _gen_rotated_shifted.flow_from_dataframe(tr_df, x_col='Class Path',
                                    y_col='Class', batch_size=batch_size,
                                    target_size=img_size, # reescalado de imagenes
                                    color_mode='grayscale', # cambio a greyscale
                                    class_mode='categorical', # clasificacion multiclase
                                  )

valid_gen_rotated_shifted = _gen_rotated_shifted.flow_from_dataframe(val_df, x_col='Class Path',
                                        y_col='Class', batch_size=batch_size,
                                        target_size=img_size, # reescalado de imagenes)
                                        color_mode='grayscale', # cambio a greyscale
                                        class_mode='categorical', # clasificacion multiclase
                                     )


# ## 3.1 Muestra de imagenes

# 1. Las imagenes se presentan con buen contraste y escala. Es importante notar que el corte de la imagen esta en diferentes orientaciones, lo que puede confundir al modelo.

classes = list(tr_gen.class_indices.keys())
images, labels = next(tr_gen)

def plot_images(images, labels, classes, title):
    plt.figure(figsize=(16, 16))
    plt.suptitle(title, fontsize=20, y=0.92)
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= 8:
            break
        plt.subplot(4,4, i + 1)
        plt.imshow(image, cmap='gray')
        class_name = classes[np.argmax(label)]
        plt.title(class_name, color='k', fontsize=15)
        plt.grid(False)

    plt.show()
plot_images(images, labels, classes, title='Ejemplos de Imágenes de Entrenamiento (Originales)')


# 2. Las imágenes rotadas ayudan a generalizar el modelo. En este caso en particular, dado que los cortes pueden presentarse en diferentes orientaciones, es probable que esto también confunda al modelo.
# 

images, labels = next(tr_gen_rotated)
plot_images(images, labels, classes, title='Ejemplos de Imágenes de Entrenamiento (Rotadas)')


# 3. En este caso, las imágenes se presentan trasladadas además de rotadas. Se puede notar rápidamente que se pierde información y, en algunos casos, se generan artefactos. Es altamente probable que esto disminuya la utilidad final del modelo.
# 

images, labels = next(tr_gen_rotated_shifted)
plot_images(images, labels, classes, title='Ejemplos de Imágenes de Entrenamiento (Rotadas y Desplazadas)')


# Se crean los modelos. Su arquitectura se puede ver mejor en la siguiente imagen: ![Arquitectura del Modelo](model.png)
# 

# Definición del modelo

epocas = 10
img_shape=(256,256,1)
  
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(256,256,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(256, kernel_size=(3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())    
    
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(Adamax(learning_rate= 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy',
                        Precision(),
                        Recall()])

model.summary()




# Cada modelo se entrena antes de crear el siguiente.
# 

hist = model.fit(tr_gen,
                     epochs=epocas,
                     validation_data=valid_gen,
                     shuffle= False)




def plot_training_history(hist):
    """ Visualiza la historia de entrenamiento del modelo.
    Args:
        hist (History): Historia de entrenamiento del modelo.
    """
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    tr_per = hist.history['precision']
    tr_recall = hist.history['recall']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    val_per = hist.history['val_precision']
    val_recall = hist.history['val_recall']

    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    index_precision = np.argmax(val_per)
    per_highest = val_per[index_precision]
    index_recall = np.argmax(val_recall)
    recall_highest = val_recall[index_recall]

    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f'Best epoch = {str(index_loss + 1)}'
    acc_label = f'Best epoch = {str(index_acc + 1)}'
    per_label = f'Best epoch = {str(index_precision + 1)}'
    recall_label = f'Best epoch = {str(index_recall + 1)}'


    plt.figure(figsize=(20, 12))
    plt.style.use('fivethirtyeight')


    plt.subplot(2, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label, zorder=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label, zorder=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(i*100)}%' for i in np.arange(0, 1.1, 0.1)])
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(Epochs, tr_per, 'r', label='Precision')
    plt.plot(Epochs, val_per, 'g', label='Validation Precision')
    plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label, zorder=2)
    plt.title('Precision and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(i*100)}%' for i in np.arange(0, 1.1, 0.1)])
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(Epochs, tr_recall, 'r', label='Recall')
    plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
    plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label, zorder=2)
    plt.title('Recall and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    #make yticks percentage
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(i*100)}%' for i in np.arange(0, 1.1, 0.1)])
    plt.legend()
    plt.grid(True)

    plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
    plt.show()


# El modelo original se estabiliza rápidamente, obteniendo pequeñas ganancias después de la cuarta iteración.
# 

plot_training_history(hist)


# Se crea el segundo modelo para las imágenes rotadas, utilizando la misma arquitectura que el modelo original.
# 

model_rotated = Sequential()

model_rotated.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(256,256,1)))
model_rotated.add(BatchNormalization())
model_rotated.add(MaxPooling2D())

model_rotated.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model_rotated.add(BatchNormalization())
model_rotated.add(MaxPooling2D())

model_rotated.add(Conv2D(256, kernel_size=(3, 3),activation='relu'))
model_rotated.add(BatchNormalization())
model_rotated.add(MaxPooling2D())    
    
model_rotated.add(Flatten())

model_rotated.add(Dense(256, activation='relu'))
model_rotated.add(Dense(128, activation='relu'))
model_rotated.add(Dense(64, activation='relu'))
model_rotated.add(Dense(4, activation='softmax'))

model_rotated.compile(Adamax(learning_rate= 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy',
                        Precision(),
                        Recall()])

model_rotated.summary()


hist_rotated = model.fit(tr_gen_rotated,
                     epochs=epocas,
                     validation_data=valid_gen_rotated,
                     shuffle= False)


plot_training_history(hist_rotated)


# En este caso, si bien el modelo muestra una tendencia clara de mejora, es fácil ver que la validación no acompaña al entrenamiento y eventualmente diverge. En una iteración futura, podría aumentarse el número de épocas y observar cómo evoluciona.
# 
# 
# 

model_rotated_shifted = Sequential()

model_rotated_shifted.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(256,256,1)))
model_rotated_shifted.add(BatchNormalization())
model_rotated_shifted.add(MaxPooling2D())

model_rotated_shifted.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model_rotated_shifted.add(BatchNormalization())
model_rotated_shifted.add(MaxPooling2D())

model_rotated_shifted.add(Conv2D(256, kernel_size=(3, 3),activation='relu'))
model_rotated_shifted.add(BatchNormalization())
model_rotated_shifted.add(MaxPooling2D())    
    
model_rotated_shifted.add(Flatten())

model_rotated_shifted.add(Dense(256, activation='relu'))
model_rotated_shifted.add(Dense(128, activation='relu'))
model_rotated_shifted.add(Dense(64, activation='relu'))
model_rotated_shifted.add(Dense(4, activation='softmax'))

model_rotated_shifted.compile(Adamax(learning_rate= 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy',
                        Precision(),
                        Recall()])

model_rotated_shifted.summary()


hist_rotated_shifted = model.fit(tr_gen_rotated_shifted,
                     epochs=epocas,
                     validation_data=valid_gen_rotated_shifted,
                     shuffle= False)


plot_training_history(hist_rotated_shifted)


# En este caso, la validación oscila con una amplitud incluso mayor que en el caso anterior, lo que permite deducir que hay una alta probabilidad de que esto haya confundido al modelo.
# 

# # 4. Tiempo de Procesamiento
# 

# Una vez entrenados los modelos, medimos el tiempo de procesamiento del conjunto de prueba en cada uno y, finalmente, obtenemos la matriz de confusión de cada modelo.
# 

# ## 4.1 Todas las imagenes

repeticiones = 10
tiempo = []
for i in range(repeticiones):
    start_time = time.time()
    pred = model.predict(ts_gen)
    end_time = time.time()
    tiempo.append({"tiempo" :end_time - start_time})

df_tiempo = pd.DataFrame(tiempo)

y_pred = np.argmax(pred, axis=1)


tiempo_rotado = []
for i in range(repeticiones):
    start_time = time.time()
    pred_rotado = model_rotated.predict(ts_gen)
    end_time = time.time()
    tiempo_rotado.append({"tiempo_rotado" :end_time - start_time})

df_tiempo_rotado = pd.DataFrame(tiempo_rotado)

y_pred_rotado = np.argmax(pred_rotado, axis=1)


tiempo_rotado_shifted = []
for i in range(repeticiones):
    start_time = time.time()
    pred_rotado_shifted = model_rotated_shifted.predict(ts_gen)
    end_time = time.time()
    tiempo_rotado_shifted.append({"tiempo_rotado_shifted" :end_time - start_time})

df_tiempo_rotado_shifted = pd.DataFrame(tiempo_rotado_shifted)

y_pred_rotado_shifted = np.argmax(pred_rotado_shifted, axis=1)


df_tiempo_final = df_tiempo.join(df_tiempo_rotado, how='outer').join(df_tiempo_rotado_shifted, how='outer')
df_tiempo_final.head(10)


# ## 4.2 Mitad de las Imagenes 

total_muestras=len(ts_gen.filenames) # total de imagenes en el generador de test
steps= int((total_muestras/ts_gen.batch_size)//2) # pasos por epoca


tiempo_mitad = []
for i in range(repeticiones):
    start_time = time.time()
    pred_mitad = model_rotated.predict(ts_gen, steps=steps) # prediccion con el modelo rotado y desplazado
    end_time = time.time()
    tiempo_mitad.append({"tiempo_mitad" :end_time - start_time})

df_tiempo_mitad = pd.DataFrame(tiempo_mitad)

y_pred_mitad = np.argmax(pred_mitad, axis=1)


tiempo_mitad_rotado = []
for i in range(repeticiones):
    start_time = time.time()
    pred_mitad_rotado = model_rotated.predict(ts_gen, steps=steps) # prediccion con el modelo rotado y desplazado
    end_time = time.time()
    tiempo_mitad_rotado.append({"tiempo_mitad_rotado" :end_time - start_time})

df_tiempo_mitad_rotado = pd.DataFrame(tiempo_mitad_rotado)

y_pred_mitad_rotado = np.argmax(pred_mitad_rotado, axis=1)


tiempo_mitad_rotado_shifted = []
for i in range(repeticiones):
    start_time = time.time()
    pred_mitad_rotado_shifted = model_rotated.predict(ts_gen, steps=steps) # prediccion con el modelo rotado y desplazado
    end_time = time.time()
    tiempo_mitad_rotado_shifted.append({"tiempo_mitad_rotado_shifted" :end_time - start_time})

df_tiempo_mitad_rotado_shifted = pd.DataFrame(tiempo_mitad_rotado_shifted)

y_pred_mitad_rotado_shifted = np.argmax(pred_mitad_rotado_shifted, axis=1)


# Se crea el dataframe final con los tiempos de cada modelo
df_tiempo_mitad_final = df_tiempo_mitad.join(df_tiempo_mitad_rotado, how='outer').join(df_tiempo_mitad_rotado_shifted, how='outer')
df_tiempo_final = df_tiempo_final.join(df_tiempo_mitad_final, how='outer')
df_tiempo_final.head(10)


# Obtenemos las medidas estadisticas de los tiempos de cada modelo 
df_tiempo_final.describe()


# Finalmente, se obtienen los resultados estadísticos del tiempo de procesamiento, donde se puede observar que, si bien la media varía, existe una relación directa entre el tiempo y la cantidad de imágenes procesadas.  
# También se puede ver que debe existir algún tipo de error en la medición, dado que hay casos donde el tiempo medido es negativo.
# Es posible que esto sea debido al uso de GPU con tensorflow, en futuras iteraciones se buscara una manera diferente de medir el tiempo corrido.
# 

# # 5. Resultados del procesamiento

# ## 5.1 Modelo Original
# 
# El modelo presenta un buen rendimiento en las clases `notumor` y `pituitary`, sin embargo, se confunde ligeramente con la clase `glioma` y no muestra sensibilidad hacia la clase `meningioma`. Es posible que, al ajustar los hiperparámetros, se logre mejorar el rendimiento del modelo en futuras iteraciones.
# 

def confusion_matrix_plot(y_predicho, title='Confusion Matrix'):
    cm = confusion_matrix(ts_gen.classes, y_predicho)
    clases= list(ts_gen.class_indices.keys())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    

confusion_matrix_plot(y_pred, title='Matriz de Confusión - Modelo Original')





from sklearn.metrics import classification_report

print(classification_report(ts_gen.classes, y_pred, target_names=ts_gen.class_indices.keys()))


# El modelo se presenta bastante robusto para la mayoria de los casos pero esta fundamentalmente atrofiado para el caso de meningioma, más rondas de preprocesamiento y tuneo de hiperparametros sera necesario antes de que pueda cumplir con su funcionamiento. 

# ## 5.2 Modelo Rotado
# 
# En este caso, a diferencia del anterior el modelo se vio completamente encandilado con la clase meningioma, asignando todas las imagenes a esta clase

confusion_matrix_plot(y_pred_rotado, title='Matriz de Confusión - Modelo Rotado')


print(classification_report(ts_gen.classes, y_pred_rotado, target_names=ts_gen.class_indices.keys()))


# ## 5.3 Modelo Rotado y Desplazado
# 
# Nuevamente se puede observar que el modelo asigna todas las imágenes de forma sobrevalorada a clases incorrectas. Esto, al igual que el modelo anterior, genera una fuerte implicancia de que este tipo de imágenes no aporta grandes beneficios al modelo con traslados y rotaciones, dado que es un ámbito donde la generalización no es tan importante como la especificidad.
# 

confusion_matrix_plot(y_pred_rotado_shifted, title='Matriz de Confusión - Modelo Rotado y Desplazado')


print(classification_report(ts_gen.classes, y_pred_rotado_shifted, target_names=ts_gen.class_indices.keys()))

