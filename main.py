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
def overfitting_model(input):
    # Create an overly complex model with many layers and neurons
    # The model has excessive capacity for the small dataset (only 600 properties)
    # Instead of learning general patterns, it will memorize individual training examples
    # No regularization is used, allowing the model to fit noise in the data
    model = tf.keras.models.Sequential([
        # Very large first layer with excessive capacity (1024 neurons)
        # This gives the model enough parameters to memorize training examples
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(input,)),
        # Many hidden layers with decreasing size but still too complex for small dataset
        # The depth allows the model to create overly specific representations
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        # Output layer: earlier layers give the final layer enough power to memorize the training data
        tf.keras.layers.Dense(1)
    ])

    # Compile with standard learning rate and very small batch size to encourage overfitting
    # Batch size of 4 means the model sees very few examples per update, making it unstable
    # Excessive capacity and instability leads to memorization of noise
    # No regularization or dropout is applied, allowing the model to overfit completely
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae']
    )

    return model

'''
Create a well-balanced model (generalize)
Args: 
    input: The number of input features
Returns:
    Compiled Keras model
'''
def optimized_model(input):
    # Create neural network model
    model = tf.keras.models.Sequential([
        # 512 neurons in this layer, use relu activation, l2 regularization, and pass how many input features to expect (input) 
        # l2 regularizer will prevent overfitting without being too aggressive (soften extreme values, prevent the model
        # from memorizing random coincidences in the data)
        tf.keras.layers.Dense(512, activation = 'relu',
                              kernel_regularizer = tf.keras.regularizers.l2(0.001), 
                              input_shape = (input,)),
        # Randomly drop 40% of neurons during training, force network to not rely on any single neuron
        tf.keras.layers.Dropout(0.4), 

        # Second hidden layer with regularization
        tf.keras.layers.Dense(256, activation = 'relu',
                              kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        # Randomly drop 30% of neurons during training 
        # (30% instead of 40% because it's working with patterns already discovered by the first layer)
        tf.keras.layers.Dropout(0.3), 

        # Hidden layers without regularization
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.2), # Randomly drop 20% of neurons during training 

        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'relu'),

        # Only 1 neuron in output layer (single price prediction for regression)
        tf.keras.layers.Dense(1)
    ])

    # Compile the model with Adam optimizer and MAE loss
    model.compile (
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002), # Lower learning rate
        loss = 'mae', # Measures average dollar difference between predictions and actual values
        metrics = ['mae'] # Track progress during training (how accurate the model actually is)
    )

    return model

'''
Train and evaluate a model with the given configuration
Args:
    model: Compiled Keras model
    x_train, y_train: Training data
    x_test, y_test: Testing data
    epochs: Number of training epochs
    batch_size: Batch size for training
    model_name: Name for display purposes
Returns:
    training_history: Training history object
    test_mae: Final test MAE
'''
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs, batch_size, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Display model architecture
    model.summary()

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{model_name} Test MAE: ${test_mae:,.2f}")

    return history, test_mae

'''
Main function to load data, train all three models, and display results
'''
def main():
    # Fix seed for reproducibility
    np.random.seed(2025)
    tf.random.set_seed(2025)

    # Load the dataset (600 samples)
    df = pd.read_csv('yyc_dataset.csv')

    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(df)

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Testing samples: {x_test.shape[0]}")
    print(f"Number of features: {x_train.shape[1]}")

    # ================================
    # Model 1: Underfitting
    # ================================
    print("\n" + "="*60)
    print("MODEL 1: UNDERFITTING")
    print("="*60)

    # Create underfitting model
    model_under = underfitting_model(x_train.shape[1])
    history_under, test_mae_under = train_and_evaluate(
        model_under, x_train, y_train, x_test, y_test,
        epochs=100, batch_size=32, model_name="Underfitting Model"
    )

    # ================================
    # Model 2: Overfitting
    # ================================
    print("\n" + "="*60)
    print("MODEL 2: OVERFITTING")
    print("="*60)

    # Create overfitting model
    model_over = overfitting_model(x_train.shape[1])
    history_over, test_mae_over = train_and_evaluate(
        model_over, x_train, y_train, x_test, y_test,
        epochs=300, batch_size=4, model_name="Overfitting Model"
    )

    # ================================
    # Model 3: Optimized
    # ================================
    print("\n" + "="*60)
    print("MODEL 3: OPTIMIZED")
    print("="*60)

    # Create optimized model
    model_opt = optimized_model(x_train.shape[1])
    history_opt, test_mae_opt = train_and_evaluate(
        model_opt, x_train, y_train, x_test, y_test,
        epochs=200, batch_size=128, model_name="Optimized Model"
    )

    # ================================
    # Summary of Results
    # ================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Underfitting Model Test MAE: ${test_mae_under:,.2f}")
    print(f"Overfitting Model Test MAE:  ${test_mae_over:,.2f}")
    print(f"Optimized Model Test MAE:    ${test_mae_opt:,.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
    