# Project_final

In this project, a complete Machine Learning project is developed. Specifically, we have focused on analysing and studying a dataset that predicts whether a song will be a hit or not. To do this, we have implemented the steps in AWS:

Step 1: We have created an instance in SageMaker of a Jupyter Notebook.

Step 2: We created a bucket where we uploaded the dataset.

Step 3: Hemos vinculado en el notebook “data-preprocessing” el dataset que hemos subido en el paso 2 y hemos realizado el data preprocessing. Seguidamente, hemos hecho el train-test-validation split. Los ficheros generados los hemos guardado de vuelta en nuestro Bucket en S3

Paso 4: hemos creado otro notebook en Jupyter llamado “model-deployment" donde importamos los datasets desde S3 a SageMaker creados en el paso anterior. Seguidamente, definimos un endpoint.

Paso 5: hemos creado una API para poder obtener los resultados de nuestras predicciones. Para poder utilizarla debemos definir una Lambda function que realizamos en el siguiente paso.

Paso 6: hemos creado una lambda function que permite crear un puente entre SageMaker y la API

Paso 7: hacemos deployment del modelo para poder lanzar las requests desde la API.
