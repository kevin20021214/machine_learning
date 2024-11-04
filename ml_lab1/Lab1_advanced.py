import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import random
# File paths
training_dataroot = 'lab1_advanced_training.csv'  # Training data file named as 'lab1_advanced_training.csv'
testing_dataroot = 'lab1_advanced_testing.csv'    # Testing data file named as 'lab1_advanced_testing.csv'
output_dataroot = 'lab1_advanced.csv'             # Output file will be named as 'lab1_advanced.csv'

# Read input CSV files into numpy arrays
training_datalist = pd.read_csv(training_dataroot).to_numpy()
testing_datalist = pd.read_csv(testing_dataroot).to_numpy()

def SplitData(data, split_ratio):
    """
    Splits the dataset into training and validation sets based on the split ratio.
    
    Parameters:
    - data (numpy.ndarray): The dataset to split.
    - split_ratio (float): The ratio of data to be used for training.
    
    Returns:
    - training_data (numpy.ndarray): Training subset.
    - validation_data (numpy.ndarray): Validation subset.
    """
    #np.random.shuffle(data)  # Shuffle data before splitting to ensure randomness
    train_size = int(len(data) * split_ratio)
    training_data = data[:train_size]
    validation_data = data[train_size:]
    return training_data, validation_data

def PreprocessData(data, is_test=False):
    """
    Preprocesses the dataset by encoding categorical variables, handling missing values,
    and removing outliers.
    
    Parameters:
    - data (numpy.ndarray): The dataset to preprocess.
    - is_test (bool): If True, the dataset is for testing and does not include the target variable.
    
    Returns:
    - preprocessed_data (numpy.ndarray): The preprocessed dataset.
    """
    if is_test:
        # Testing data has 7 features (no 'gripForce')
        df = pd.DataFrame(data, columns=['age', 'gender', 'height', 'weight', 'bodyFat', 'diastolic', 'systolic'])
    else:
        # Training and validation data have 8 columns (including 'gripForce')
        df = pd.DataFrame(data, columns=['age', 'gender', 'height', 'weight', 'bodyFat', 'diastolic', 'systolic', 'gripForce'])

    # Encode 'gender' column: 'F' -> 0, 'M' -> 1
    df['gender'] = df['gender'].map({'F': 1, 'M': 5})
    # Handle missing values by filling NaNs with the median of each column
    df = df.dropna()
    #df = df.fillna(df.median())
    df = df.astype('float64')
    # Remove outliers using the Interquartile Range (IQR) method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # Keep rows where all features are within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    if is_test == False :df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df.to_numpy()

def Regression(dataset):
    """
    Performs linear regression on the training dataset.
    
    Parameters:
    - dataset (numpy.ndarray): The preprocessed training dataset, including features and target.
    
    Returns:
    - w (numpy.ndarray): The learned weights of the regression model.
    """
    X = dataset[:, :-1]  # Features (7 columns)
    y = dataset[:, -1]   # Target variable ('gripForce')

    # Initialize weights (7 weights for 7 features)
    num_dimensions = X.shape[1]
    w = np.zeros(num_dimensions)

    # Hyperparameters
    num_iteration = 600000
    learning_rate = 0.000033
    m = len(y)

    for iteration in range(num_iteration):
        y_pred = X.dot(w)  # Predicted values
        error = y_pred - y  # Errors

        # Compute cost (Mean Squared Error)
        cost = (1/(2*m)) * np.sum(error ** 2)

        # Compute gradient
        gradient = (1/m) * X.T.dot(error)

        # Update weights
        w -= learning_rate * gradient

        # Print cost every 100 iterations for monitoring
        if iteration % 100000 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")

    return w

def MakePrediction(w, test_dataset):
    """
    Uses the regression model to make predictions on the test dataset.
    
    Parameters:
    - w (numpy.ndarray): The learned weights of the regression model.
    - test_dataset (numpy.ndarray): The preprocessed test dataset (7 features).
    
    Returns:
    - prediction (numpy.ndarray): The predicted 'gripForce' values.
    """
    X_test = test_dataset  # Test data has 7 features
    # Perform prediction
    prediction = X_test.dot(w)
    return prediction

# (1) Split data into training and validation sets
training_data, validation_data = SplitData(training_datalist, 0.8)

# (2) Preprocess the training and validation data
training_data = PreprocessData(training_data, is_test=False)
validation_data = PreprocessData(validation_data, is_test=False)  # Includes 'gripForce' for MAPE calculation

# (3) Train the regression model
w = Regression(training_data)

# (4) Predict on the validation dataset and calculate MAPE
validation_features = validation_data[:, :-1]  # Extract features (7 columns)
validation_true = validation_data[:, -1]      # Extract true 'gripForce' values

# Make predictions on validation features
validation_pred = MakePrediction(w, validation_features)

# Calculate Mean Absolute Percentage Error (MAPE)
# Handle potential division by zero by replacing zeros in y_true with a small number (1e-6)
non_zero_y_true = np.where(validation_true == 0, 1e-6, validation_true)
MAPE = np.mean(np.abs((validation_true - validation_pred) / non_zero_y_true)) * 100
print(f"Validation MAPE: {MAPE}%")

# (5) Preprocess the testing data and make predictions
testing_data = PreprocessData(testing_datalist, is_test=True)  # Test data has 7 features
output_datalist = MakePrediction(w, testing_data)+np.mean(validation_true-validation_pred)

# Write the predictions to the output CSV file
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'gripForce'])
    for i in range(len(output_datalist)):
        writer.writerow([i, output_datalist[i]])
