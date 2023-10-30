# Proyecto de Clasificación de Transacciones Fraudulentas con AutoML 

## Descripción
Este proyecto se centra en la detección de transacciones fraudulentas mediante el uso de aprendizaje automático (AutoML) y aplicando Tuning de hiperparámetros. Utilizamos un dataset de transacciones financieras y aplicamos un proceso de clasificación para identificar transacciones fraudulentas.

Abarcaremos la creacion de un modelo de clasificacion aplicando el El algoritmo kNN (K-Nearest Neighborgs) si bien este algoritmo es usado en modelos de regresion y clasificacion en este proyecto lo utilizaremos para clasificacion.

## Modelo de Clasificación: K-Nearest Neighbors (KNN)

En este proyecto, hemos utilizado el algoritmo K-Nearest Neighbors (KNN) para abordar el problema de clasificación de transacciones fraudulentas. KNN es un algoritmo de aprendizaje supervisado que se utiliza comúnmente en problemas de clasificación y agrupación.

### ¿Cómo funciona KNN?

KNN es un algoritmo simple pero efectivo. Funciona asignando una etiqueta de clase a un punto de datos basada en la mayoría de las etiquetas de clase de sus vecinos más cercanos. En otras palabras, clasifica un punto de datos según la clase predominante entre sus k vecinos más cercanos.

### Ventajas de KNN

- Fácil de entender y aplicar.
- No requiere supuestos sobre la distribución de los datos.
- Puede ser utilizado tanto para problemas de clasificación como de regresión.
- Puede manejar datos con múltiples características (dimensiones).

### Uso en este Proyecto

Hemos entrenado un modelo KNN utilizando Azure AutoML para identificar transacciones fraudulentas en nuestro conjunto de datos. El mejor modelo encontrado por AutoML se basa en el algoritmo KNN, lo que demuestra su eficacia en esta tarea de clasificación.

Para obtener más detalles sobre la configuración y los resultados del modelo KNN en este proyecto, consulte el notebook 'orchestrator.ipynb'.


## Requisitos Mínimos para el despliegue del proyecto:
Cuenta de Azure: Debes tener una cuenta en Microsoft Azure. Si no tienes una cuenta, puedes registrarte en Microsoft Azure.

Azure Machine Learning: Debes tener una instancia de Azure Machine Learning configurada. Puedes crear una en el portal de Azure.


## Pasos del Proyecto
### 1. Preparación de Datos (data-preprocessing)
En el Jupyter Notebook `data-preprocessing.ipynb`, realizamos las siguientes tareas:

- Exploración del dataset: Analizamos la distribución de las columnas y contamos las transacciones por clase.
- Limpieza de datos: Eliminamos filas con valores vacíos y preprocesamos los datos.
- Documentación: Explicamos por qué elegimos este dataset, el problema que resolvemos y una descripción detallada del dataset.

### 2. Entrenamiento de Modelos con AutoML (orchestrator)
En el Jupyter Notebook `orchestrator.ipynb`, llevamos a cabo lo siguiente:

- Separación de datos: Dividimos el conjunto de datos en entrenamiento y validación.
- Creación de un cluster de cómputo en Azure: Justificamos la elección de la infraestructura.
- Configuración de un job AutoML: Entrenamos uno o varios modelos utilizando el cluster.
- Análisis de resultados: Recuperamos información sobre los modelos probados, sus hiperparámetros y la precisión del mejor modelo.
- Creación de un endpoint: Registramos y desplegamos el mejor modelo en un endpoint, explicando la configuración utilizada.
- Realización de predicciones: Mostramos cómo realizar inferencias desde el endpoint.

### 3. Tuning de Hiperparámetros (orchestrator-hp)
En el Jupyter Notebook `orchestrator-hp.ipynb`, llevamos a cabo lo siguiente:

- Selección de modelos: Elegimos al menos dos modelos para afinar.
- Tuning de hiperparámetros: Seleccionamos al menos dos hiperparámetros por modelo, especificamos los valores posibles y configuramos el job de tuning.
-Finalmente Comentamos sobre los modelos seleccionados, los hiperparámetros y los resultados en comparación con el modelo base.

## Datos del Proyecto

### Dataset

El dataset utilizado en este proyecto debe descargarse y colocarse en el mismo directorio que los notebooks y los scripts. Puedes descargar el dataset desde el siguiente enlace: [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/].

Esta estructura permitirá que los notebooks accedan a los datos de manera adecuada. Si tienes alguna pregunta o necesita ayuda con la configuración, no dudes en ponerte en contacto con nosotros.


## Estructura de Directorios

El repositorio de Git contiene los siguientes archivos y directorios clave:

- `data-preprocessing.ipynb`: Un Jupyter Notebook para la preparación de datos, que incluye la exploración del dataset, limpieza de datos y documentación.

- `orchestrator.ipynb`: Un Jupyter Notebook para el proceso de AutoML, que incluye la separación de datos, creación de clusters de cómputo y el entrenamiento de modelos.

- `orchestrator-hp.ipynb`: Jupyter Notebook para el afinamiento de hiperparámetros, que permite ajustar los modelos y parámetros seleccionados.



## Contacto
Para preguntas o más información, contáctanos a través de correo electrónico en belenfc01@gmail.com.

## Historial de Cambios
- Versión 1.0.0 (29/10/2023): Primera versión del proyecto.

