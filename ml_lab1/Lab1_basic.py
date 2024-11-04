import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import random
training_dataroot = 'lab1_basic_training.csv' # Training data file file named as 'lab1_basic_training.csv'
testing_dataroot = 'lab1_basic_testing.csv'   # Testing data file named as 'lab1_basic_testing.csv'
output_dataroot = 'lab1_basic.csv' # Output file will be named as 'lab1_basic.csv'

training_datalist =  [] # Training datalist, saved as numpy array
testing_datalist =  [] # Testing datalist, saved as numpy array

output_datalist =  [] # Your prediction, should be a list with 100 elements

# Read input csv to datalist
with open(training_dataroot, newline='') as csvfile:
  training_datalist = pd.read_csv(training_dataroot).to_numpy()

with open(testing_dataroot, newline='') as csvfile:
  testing_datalist = pd.read_csv(testing_dataroot).to_numpy()

def SplitData(data, split_ratio):
  """
  Splits the given dataset into training and validation sets based on the specified split ratio.
  Parameters:
  - data (numpy.ndarray): The dataset to be split. It is expected to be a 2D array where each row represents a data point and each column represents a feature.
  - split_ratio (float): The ratio of the data to be used for training. For example, a value of 0.8 means 80% of the data will be used for training and the remaining 20% for validation.
  Returns:
  - training_data (numpy.ndarray): The portion of the dataset used for training.
  - validation_data (numpy.ndarray): The portion of the dataset used for validation.
  """
  training_data = []
  validation_data = []
  # TODO
  #np.random.shuffle(data)# random
  train_size = int(len(data) * split_ratio) #len of train data
  training_data = data[:train_size]
  validation_data = data[train_size:]
  return training_data, validation_data

def PreprocessData(data):
    """
    Preprocess the given dataset and return the result.

    Parameters:
    - data (numpy.ndarray): The dataset to preprocess. It is expected to be a 2D array where each row represents a data point and each column represents a feature.

    Returns:
    - preprocessedData (numpy.ndarray): Preprocessed data.
    """
    preprocessedData = []
    # TODO
    df = pd.DataFrame(data)

    # 處理 NaN 值 (用中位數填補)
    df = df.fillna(df.median())

    # 檢測和移除異常值（使用IQR）
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    preprocessedData = df.to_numpy()
    return preprocessedData

def Regression(dataset):
    """
    Performs regression on the given dataset and return the coefficients.

    Parameters:
    - dataset (numpy.ndarray): A 2D array where each row represents a data point.

    Returns:
    - w (numpy.ndarray): The coefficients of the regression model. For example, y = w[0] + w[1] * x + w[2] * x^2 + ...
    """

    X = dataset[:, :1]
    y = dataset[:, 1]
    # TODO: Decide on the degree of the polynomial
    degree = 3  # For example, quadratic regression

    # Add polynomial features to X
    X_poly = np.ones((X.shape[0], 1))  # Add intercept term (column of ones)
    print(X_poly.shape)
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))  # Add x^d terms to feature matrix
    # Initialize coefficients (weights) to zero
    num_dimensions = X_poly.shape[1]  # Number of features (including intercept and polynomial terms)
    w = np.zeros(num_dimensions)
    # TODO: Set hyperparameters
    num_iteration = 10000
    learning_rate = 0.0000000000001   # Gradient Descent
    m = len(y)  # Number of data points
    
    for iteration in range(num_iteration):
        # TODO: Prediction using current weights and compute error
        y_pred = X_poly.dot(w)
        error = y_pred - y
        # TODO: Compute gradient
        cost = (1/(2*m)) * np.sum(error ** 2)
        gradient = (1/m) * X_poly.T.dot(error)
        # TODO: Update the weights
        w -= learning_rate * gradient
        # TODO: Optionally, print the cost every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")
    return w

def MakePrediction(w, test_dataset):
    """
    Predicts the output for a given test dataset using a regression model.

    Parameters:
    - w (numpy.ndarray): The coefficients of the model, where each element corresponds to
                               a coefficient for the respective power of the independent variable.
    - test_dataset (numpy.ndarray): A 1D array containing the input values (independent variable)
                                          for which predictions are to be made.

    Returns:
    - list/numpy.ndarray: A list or 1d array of predicted values corresponding to each input value in the test dataset.
    """
    prediction = []
    # TODO
    X_test = test_dataset[:, 0].reshape(-1, 1)
    degree = 3
    X_poly_test = np.ones((X_test.shape[0], 1))
    for d in range(1, degree + 1):
        X_poly_test = np.hstack((X_poly_test, X_test ** d))

    prediction = X_poly_test.dot(w)
    return prediction
# TODO

# (1) Split data
training_data, validation_data = SplitData(training_datalist, 0.8)
# (2) Preprocess data
training_data = PreprocessData(training_data)
validation_data = PreprocessData(validation_data)
# (3) Train regression model
w = Regression(training_data)
# (4) Predict validation dataset's answer, calculate MAPE comparing to the ground truth
validation_pred = MakePrediction(w, validation_data)
y_true = validation_data[:, 1]
MAPE = np.mean(np.abs((y_true - validation_pred) / y_true)) * 100
print(f"Validation MAPE: {MAPE}%")
# (5) Make prediction of testing dataset and store the values in output_datalist
testing_data = PreprocessData(testing_datalist)
output_datalist = MakePrediction(w, testing_data)
# Assume that output_datalist is a list (or 1d array) with length = 100
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['Id', 'gripForce'])
  for i in range(len(output_datalist)):
    writer.writerow([i,output_datalist[i]])
