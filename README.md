# Project_final

This project involves the development of a comprehensive Machine Learning pipeline. The main focus was on exploring and analyzing a dataset to predict whether a song will become a hit or not. The following steps were implemented using AWS:

Step 1: A SageMaker instance of Jupyter Notebook was created.

Step 2: A bucket was set up to store the dataset.

Step 3: In the "data-preprocessing" notebook, we linked the dataset uploaded in step 2 and performed the necessary data preprocessing. This included a train-test-validation split, with the resulting files being saved back to our S3 bucket.

Step 4: Another notebook was created called "model-deployment". Here, we imported the datasets from S3 to SageMaker and defined an endpoint.

Step 5: An API was created to obtain the prediction results. A Lambda function was defined in the next step to enable the use of this API.

Step 6: A Lambda function was created to serve as a bridge between SageMaker and the API.

Step 7: The model was deployed to allow for API requests to be launched.
