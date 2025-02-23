## Summative_Plant_Diseases
## Problem Staetment 
#### This project leverages technology and education to create awareness about plant diseases and empower small-scale farmers in rural Kenya with effective detection and management solutions. A machine learning model using Convolutional Neural Networks (CNNs) will be implemented to detect plant diseases from uploaded images. Farmers will receive real-time diagnoses, treatment recommendations, and access to educational modules on AI-driven pest and disease management.
## Dataset  
#### The dataset I used was a **plant diseases dataset**. It is used to predict and categorize the images based on the disease and class.  
#### https://www.kaggle.com/datasets/puritykihiu/plant-dataset
## Table of Model Comparisons
| Train Instance | Optimizer used (Adam, RMSProp) | Regularizer Used (L1 and L2) | Epochs | Early Stopping (Yes or No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|---------------|--------------------------------|------------------------------|--------|----------------------------|------------------|--------------|----------|----------|--------|-----------|
| Instance 1   | Adam                           | None                         | 10   | None                       | 7            | None         | 0.681    | 0.576    | 0.557  | 0.597     |
| Instance 2   | RMSProp                        | L2                           | Yes    | 20%                        | 7          | None         | 0.685    | 0.5849   | 0.849  | 0.4503    |
| Instance 3   | Batch Normalization            | Adam                         | Yes    | 0.2                         | 7           | None         | 0.680    | 0.480    | 0.380  | 0.680     |
| Instance 4   | L2                             | Adam                         | No     | 10%                         | 7            | None         |          |          |        |           |
| Instance 5   | L1 + L2                        | RMSProp                      | Yes    | 15%                         | 7           | None         |          |          |        |           |
## Discussion

## Which Model Performed is the best?
