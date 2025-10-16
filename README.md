Email Spam Detection Project


This project implements a deep learning model using Bidirectional LSTMs to classify email messages as either "spam" or "ham" (not spam).


Project Overview

The goal of this project is to build a robust and accurate spam detection system for email messages. The project utilizes a dataset of email messages, preprocesses the text data, trains a Bidirectional LSTM model, and evaluates its performance using various metrics.


Dataset

The dataset used in this project is spam.csv, which contains email messages labeled as either "spam" or "ham".


Preprocessing

The text data undergoes the following preprocessing steps:
Removal of punctuation
Conversion to lowercase
Tokenization
Removal of stop words
Stemming
Lemmatization
After preprocessing, the text data is converted into numerical sequences using tokenization and then padded to ensure uniform input length for the LSTM model.


Model Architecture

The model is a sequential Keras model consisting of:
An Embedding layer to represent words as dense vectors.
Two Bidirectional LSTM layers to capture sequential dependencies in both directions.
A Dense layer with ReLU activation.
A Dropout layer for regularization.
A final Dense layer with sigmoid activation for binary classification.
Training and Evaluation
The model is compiled with the Adam optimizer and Binary Crossentropy loss with label smoothing. Early stopping is used to prevent overfitting. The model is trained on a split of the dataset and evaluated on a separate test set.


Evaluation metrics include:

Accuracy
Confusion Matrix
Precision, Recall, and F1-score
ROC Curve and AUC score
Classification Report
Dependencies


The project requires the following libraries:

pandas
numpy
tensorflow
scikit-learn
nltk
matplotlib
seaborn
Usage
Load the dataset: Ensure the spam.csv file is in the correct directory or provide the correct path.
Run the notebook cells: Execute the code cells in the notebook sequentially to perform data loading, preprocessing, model training, and evaluation.
Predict on new samples: Use the predict_sample function to classify new email messages.
Results
The model achieves good performance in classifying spam messages, as indicated by the evaluation metrics. The confusion matrix, classification report, and ROC curve provide detailed insights into the model's performance.
