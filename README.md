## Summative_Plant_Diseases
## Problem Statement 
### This project leverages technology and education to create awareness about plant diseases and empower small-scale farmers in rural Kenya with effective detection and management solutions. A machine learning model using Convolutional Neural Networks (CNNs) will be implemented to detect plant diseases from uploaded images. Farmers will receive real-time diagnoses, treatment recommendations, and access to educational modules on AI-driven pest and disease management.
## Saved Models
### The files were too large to be pushed to the github and I saved them in a Google drive  here is the link:
https://drive.google.com/drive/folders/1PBcSEyPHlyn3PEEmJr61yqWufLc7KB6k?usp=drive_link
![image](https://github.com/user-attachments/assets/0ac7bb22-cd1b-49b1-ba48-1747ff59d5fe)

## Dataset  
### The dataset I used was a **plant diseases dataset**. It is used to predict and categorize the images based on the disease and class.  
### https://www.kaggle.com/datasets/puritykihiu/plant-dataset
### The dataset consists of images of plants, each associated with one of 38 classes representing different plant diseases. The dataset has been split into:

### Training Set:  56251 images
### Validation Set: 14044 images
### Test Set: 654 Images
### Data Preprocessing
### Images are resized to 128x128.
### Data augmentation is applied to improve model generalization.
## Table of Model Comparisons
| Train Instance | Model Name | Optimizer Used | Regularizer Used (L1, L2) | Epochs | Early Stopping | Number of Layers | Dropout | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|---------------|------------|----------------|---------------------------|--------|---------------|------------------|---------|--------------|----------|----------|--------|-----------|
| Instance 1   | Model 1    | Adam           | None                      | 10     | No            | 7 (3 Conv2D, 3 MaxPooling, 1 Dense) | 0.0     | None         | 0.898    | 0.4954   | 0.4954  | 0.4954    |
| Instance 2   | Model 2    | RMSProp        | L2(0.05)                   | 10     | Yes           | 7 (3 Conv2D, 3 MaxPooling, 1 Dense) | 0.5     | 0.01         | 0.780    | 0.3945   | 0.3945  | 0.3945    |
| Instance 3   | Model 3    | Nadam          | L1(0.05)                   | 10     | No            | 7 (3 Conv2D, 3 MaxPooling, 1 Dense) | 0.05    | 0.001        | 0.843    | 0.4817   | 0.4817  | 0.4817    |
| Instance 4   | Model 4    | SGD            | L2(0.005)                  | 10     | Yes           | 7 (3 Conv2D, 3 MaxPooling, 1 Dense) | 0.5     | 0.01         | 0.939    | 0.5015   | 0.5015  | 0.5015    |
| Instance 5   | Model 5    | Adam           | L2(0.005)                  | 15     | Yes           | 7 (3 Conv2D, 3 MaxPooling, 1 Dense) | 0.5     | 0.001        | 0.887    | 0.4878   | 0.4878  | 0.4878    |

## Discussion
#### When I started creating my model at first, it was overfitting and its Accuracy was at 0.97. The f1 score was 0.2 which meant my model was overfitting. After doing more than different training I came to discover that the data was the one having an issue. The testing set was having a 1 class instead  of the 38 classes that the other train and validate had, I had to make a new dataset to ensure they had the same classes. The new dataset worked best where the best model was Model4( model_4 = define_model('SGD',l2(0.005), True, 0.5,0.01). 
### The Best model I used was the SGD optimizer which is the best model to optimise large datasets in ML, other than the Adam optimizer it adjusts the running rate for parameters separately while SGD combines all the parameters.
### With the combination of the regularization techniques (L2 and dropout) which reduced overfitting.
### Instance one (Adam), The model was having a low F1, recall and precision because I hadn't included any regularization parameters. Without dropout or L2/L1 regularization,they were no parameters and the also the model was learning the dataset.
### Instance two(RMSprop) I decided to use RMSprop but saw a significant drop in performance to 0.780 accuracy. It would have been caused  by too aggressive L2 regularization (0.05) along with a high dropout rate (0.5). Looking at the metrics (F1, recall, precision all at 0.3945), I could see these strong regularization parameters were actually causing underfitting.
### Inance three (Nadma), Nadam optimizer where it is a combination of both adam and Nesterov momentum achieved 0.843 accuracy. I used L1 regularization at 0.05 for feature selection with a lower dropout rate of 0.05 where it lead to a lower metrics (F1, recall, precision at 0.4817).
### instance Four (SGD)Using SGD an where it used to large dataset where it lead best accuracy of 0.939. I discovered that using a lighter L2 regularization (0.005) combined with dropout (0.5) worked perfectly. What really made this model good was how I balanced the learning rate at 0.01 - not too aggressive, not too conservative. 
### Instnce Five (Adam, Dropout), using of Adam more paratermers I tought it would have a high accuracy than SGD since adam is considered to the best optimizer. Dispite me using more epochs to 15 and got an accuracy of 0.887. I kept the same L2 regularization (0.005) that worked well in Instance Four, but I found that even with a lower learning rate (0.001) and early stopping it went lower. While the metrics were consistent (0.4878 across F1, recall, and precision),
## Comparison of the Neural Network and an ML algorithm
### The Neral network is the best as it has a test accuracy of 0.939 while the SVM has 0.754. The parameter for model 4 was SGD and the parameters were combined and  CNNs work better for images because they learn patterns automatically, while SVMs need manual feature selection. However, SVMs are simpler to use and work well with small datasets. Model 4 performed best by using dropout and L2 regularization to prevent overfitting.
## Video Presentation
### https://youtu.be/s3vui_sD0QY
## Way to run my model and the saved one 
#### ├── notebooks/
#### │   ├── main_notebook.ipynb
#### │   └── utils/
#### │       ├── data_preprocessing.py
#### │       ├── model_evaluation.py
#### │       └── visualization.py
#### ├── models/
#### │   ├── model_1.keras
#### │   ├── model_2.keras
#### │   ├── model_3.keras
#### │   ├── model_4.keras
#### │   ├── model_5.keras
#### │   └── svm_model.pkl
#### ├── data/
#### │   ├── raw/
#### │       ├── train_images/
#### │       └── test_images/
#### │   ├── processed/
#### │       ├── train_data.npy
#### │       └── test_data.npy
#### ├── assets/
#### │   └── images/
#### │       ├── accuracy_plot.png
#### │       └── confusion_matrix.png
#### ├── requirements.txt
#### └── README.md


