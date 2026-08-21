'''
Name: Anna Khoriakova
Date: 2026-08-21
Description: This program demonstrates neural network models for property value prediction 
             using the City of Calgary property assessment dataset.
             Includes examples of underfitting, overfitting, and an optimized model.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

'''
Preprocess the dataset for neural network training
Args: 
    df: pandas DataFrame with the property data
Returns:
    x_train, x_test, y_train, y_test: Preprocessed and split datasets
'''
def preprocess_data(df):
    data = df.copy()  # Create copy to avoid modifying original data
    target = data['ASSESSED_VALUE'].values  # Store target variable

    # Remove columns that will not be used as features
    columns_to_remove = [
        'ADDRESS',
        'COMM_NAME',
        'ASSESSMENT_CLASS_DESCRIPTION',
        'LAND_SIZE_SF',
        'ASSESSED_VALUE'
    ]

    # Only remove columns that exist in the dataframe
    data = data.drop([column for column in columns_to_remove if column in data.columns], axis=1)

    # Convert categorical text data to a binary format (one-hot encoding)
    categorical_columns = ['COMM_CODE', 'ASSESSMENT_CLASS', 'PROPERTY_TYPE']
    data = pd.get_dummies(data, columns=categorical_columns)

    # Convert features and target to numpy arrays
    features = data.values.astype(np.float32)
    target = target.astype(np.float32)

    # Split data into training and testing sets (5:1 ratio)
    n_samples = features.shape[0]
    n_train = int(n_samples * (5/6))

    # Create a randomly shuffled list of indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    x_train = features[train_indices]
    x_test = features[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]

    # Normalize each feature column
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    # Safety check for features with no variation
    train_std = np.where(train_std == 0, 1, train_std)

    # Standardize features (mean=0, std=1)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    return x_train, x_test, y_train, y_test

'''
Create a model designed to underfit the data
Args: 
    input: The number of input features
Returns:
    Compiled Keras model
'''
def underfitting_model(input):
    # Create a simple neural network with minimal capacity
    # The model is too simple to capture the complex relationships in the data
    model = tf.keras.models.Sequential([
        # Only 2 neurons in the hidden layer (too few to learn complex patterns)
        # With only 2 neurons, the model can only learn very basic patterns like "bigger houses cost more"
        # It cannot capture how location, year of construction, and property type combine to affect value
        # No regularization is applied to keep the model as simple as possible
        tf.keras.layers.Dense(2, activation='relu', input_shape=(input,)),
        # Single neuron output layer for regression (predicts one value: the property price)
        tf.keras.layers.Dense(1)
    ])

    # Compile with standard learning rate
    # Using MAE (Mean Absolute Error) as the loss function
    # MAE measures the average dollar difference between predictions and actual values
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae']
    )

    return model

'''
Create a model designed to overfit the data
Args: 
    input: The number of input features
Returns:
    Compiled Keras model
'''


'''
Create a well-balanced model (generalize)
Args: 
    input: The number of input features
Returns:
    Compiled Keras model
'''
