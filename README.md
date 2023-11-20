# ID223Lab1
This repository is Lab 1 for ID2223 Scalable ML and DL in KTH. It based on [Iris Serverless ML System](https://github.com/ID2223KTH/id2223kth.github.io/tree/master/src/serverless-ml-intro)
from the course repository. The code in this repo only shows the code of Task2. The code of Task1 can be found in the previous link.
This lab is the first lab for ID2223 Scalable Machine Learning (ML) and Deep Learning (DL) in KTH. It contains two tasks. 
Task 1 is about using the code about iris flower identification provided by the course to learn how to realize the serverless ML systems. 
Task 2 is about building a serverless ML system for the wine quality dataset to realize the wine quality prediction function.

## ID2223Lab1Task1
In this task, we follow the lab tutorial. By doing this task, we learned the knowledge about how to use websites and tools like hopsworks.ai, modal.com, huggingface.com etc.
This task has three sub-tasks.
1. Build and run iris-eda-and-backfill-feature-group.ipynb to make use the features from hopsworks.ai. Build and run the feature pipeline iris-feature-daily-pipeline on Modal.
2. Run the training pipeline iris-training-pipeline.ipynb using the data (features, labels) from hopsworks.ai and save the model to hopsworks.ai.
3. Build and run the inference pipeline iris-batch-inference-pipeline.py using the data from hopsworks.ai and send the predictions to hopsworks.ai. Then build a Gradio UI (Iris UI + Iris Monitor UI) on Hugging Face Spaces.
By doing this task, we learned the knowledge about how to use these tools and laid the foundation of doing Task2.

## ID2223Lab1Task2
In this task, we use the similar method as Task1 to build a Serverless ML system for the Wine Quality Dataset to predict the quality of the wine.

### How to Run
1. First, using Jupyter Notebook to run wine-eda-and-backfill-feature-group.ipynb. Then, run daily-wine-feature-pipeline.py.
2. Then, using Jupyter Notebook to run wine-training-pipeline.ipynb to train the modal.
3. Then, run wine-batch-inference-pipeline.py to get the predictions.
4. run app.py in huggingface-spaces-wine and huggingface-spaces-wine-monitor folders generate the URL of the GUI

### Implementation
The implementation includes 4 source code files. The description of the source code files is as follows.

#### wine-eda-and-backfill-feature-group.ipynb
This file cleans the data in *wine.csv* and uploads the features to a feature group in hopsworks. 
In terms of data cleaning, we first fill the missing data with a category corresponding to "Unknown". Then, we transform categorical variable (in this case : type) into numerical variables.
After that, we will drop the columns that don't have predictive power by analyzing the correlation matrix.
After the data cleaning, we write the features to the feature store in hopsworks.ai as a Feature Group.

#### wine-feature-pipeline-daily.py
This file is deployed on cloud by using [Modal](https://modal.com/). It generates a new wine to the dataframe every day.

#### wine-training-pipeline.ipynb
This file trains the ML model and then upload the model to the hopsworks model registry.

#### wine-batch-inference-pipeline.py
This file get data from feature store and make the prediction and upload the prediction results to hopsworks.ai. 

### public URL for 2 GUI
Wine quality prediction: 
Wine quality prediction monitor: 

## Contributors
The contributors of this repo are Tianyu Deng and Wenjian Ma.
