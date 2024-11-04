import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import random

# File paths
training_dataroot = 'lab1_advanced_training.csv'
testing_dataroot = 'lab1_advanced_testing.csv'
output_dataroot = 'lab1_advaned.csv'

# Read input CSV files into numpy arrays
training_datalist = pd.read_csv(training_dataroot).to_numpy()
testing_datalist = pd.read_csv(testing_dataroot).to_numpy()

def SplitData(data, split_ratio):
    train_size = int(len(data) * split_ratio)
    training_data = data[:train_size]
    validation_data = data[train_size:]
    return training_data, validation_data

def PreprocessData(data, is_test=False):
    if is_test:
        df = pd.DataFrame(data, columns=['age', 'gender', 'height', 'weight', 'bodyFat', 'diastolic', 'systolic'])
    else:
        df = pd.DataFrame(data, columns=['age', 'gender', 'height', 'weight', 'bodyFat', 'diastolic', 'systolic', 'gripForce'])

    df['gender'] = df['gender'].map({'F': 5, 'M': 7})
    df = df.dropna()
    df = df.astype('float64')

    # Remove outliers using the Interquartile Range (IQR) method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    if not is_test:
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Standardize the features (mean = 0, std = 1)
    if not is_test:
        mean = df.iloc[:, :-1].mean()
        std = df.iloc[:, :-1].std()
        df.iloc[:, :-1] = (df.iloc[:, :-1] - mean) / std
    else:
        df = (df - df.mean()) / df.std()

    return df.to_numpy()

def add_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                new_feature = (X[:, i] * X[:, j]) ** d
                new_feature = new_feature.reshape(-1, 1)
                X_poly = np.hstack((X_poly, new_feature))
    return X_poly

def Regression(dataset, degree=2, lambda_reg=0.3, learning_rate=0.00005, num_iteration=100000):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_poly = add_polynomial_features(X, degree)
    num_dimensions = X_poly.shape[1]
    w = np.random.randn(num_dimensions) * 0.01

    m = len(y)

    for iteration in range(num_iteration):
        y_pred = X_poly.dot(w)
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error ** 2) + (lambda_reg/(2*m)) * np.sum(w**2)
        gradient = (1/m) * X_poly.T.dot(error) + (lambda_reg/m) * w
        w -= learning_rate * gradient

        if iteration % 5000 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")

    return w

def MakePrediction(w, test_dataset, degree=2):
    X_poly_test = add_polynomial_features(test_dataset, degree)
    prediction = X_poly_test.dot(w)
    return prediction

def cross_validation(data, k_folds=1, degree=2, lambda_reg=0.3, learning_rate=0.00005, num_iteration=100000):
    data_split = np.array_split(data, k_folds)
    best_w = None
    best_MAPE = float('inf')

    for fold in range(k_folds):
        train_data = np.vstack([data_split[i] for i in range(k_folds) if i != fold])
        val_data = data_split[fold]

        w = Regression(train_data, degree, lambda_reg, learning_rate, num_iteration)
        val_features = val_data[:, :-1]
        val_true = val_data[:, -1]
        val_pred = MakePrediction(w, val_features, degree)

        non_zero_y_true = np.where(val_true == 0, 1e-6, val_true)
        MAPE = np.mean(np.abs((val_true - val_pred) / non_zero_y_true)) * 100

        if MAPE < best_MAPE:
            best_MAPE = MAPE
            best_w = w

    print(f"Best MAPE from cross-validation: {best_MAPE}%")
    return best_w

# (1) Split data into training and validation sets
training_data, validation_data = SplitData(training_datalist, 0.8)

# (2) Preprocess the training and validation data
training_data = PreprocessData(training_data, is_test=False)
validation_data = PreprocessData(validation_data, is_test=False)

# (3) 使用自行實現的交叉驗證選擇最佳模型
best_w = cross_validation(training_data, k_folds=10, degree=3, lambda_reg=0.05, learning_rate=0.00005, num_iteration=500000)

# (4) 使用最佳權重進行測試集預測
testing_data = PreprocessData(testing_datalist, is_test=True)
output_datalist = MakePrediction(best_w, testing_data, degree=3) + np.mean(validation_data[:, -1] - MakePrediction(best_w, validation_data[:, :-1], degree=3))

# (5) Write the predictions to the output CSV file
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'gripForce'])
    for i in range(len(output_datalist)):
        writer.writerow([i, output_datalist[i]])
