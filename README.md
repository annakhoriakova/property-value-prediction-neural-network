# Property Value Prediction Using Neural Networks

Neural network models for predicting property assessed values using City of Calgary property assessment data. This project demonstrates three different modeling approaches: underfitting, overfitting, and an optimized model.

## Project Structure
```
property-value-prediction-neural-network/
├── main.py              # Main script with all models and training logic
├── yyc_dataset.csv      # City of Calgary property dataset
└── README.md            # Current file
```

## Dataset

The dataset contains property assessment information including:

- COMM_CODE: Community code (categorical)
- YEAR_OF_CONSTRUCTION: Year the property was built
- PROPERTY_TYPE: Land only or land with improvements
- LAT / LON: Geographic coordinates
- LAND_SIZE_SM: Land area in square meters
- ASSESSMENT_CLASS: Property classification (residential, commercial, etc.)
- ASSESSED_VALUE: Target variable (property value)

The dataset used is a 600-sample subset of the City of Calgary property assessment data.

## Models

Three neural network models are implemented to demonstrate different learning behaviors:

1. **Underfitting Model**: A simple network with only 2 neurons in the hidden layer. Intentionally too simple to capture complex patterns in the data.

2. **Overfitting Model**: An overly complex network with 8 layers (1024→512→256→128→64→32→16→1). Designed to memorize training data and perform poorly on new data.

3. **Optimized Model**: A balanced architecture with dropout and L2 regularization. Uses 6 layers (512→256→128→64→32→1) with regularization techniques to prevent overfitting while maintaining good performance.

## Requirements

- Python 3.13
- TensorFlow 2.20
- Pandas
- NumPy

## Installation

To run the property value prediction models locally, install the required dependencies and prepare the dataset.

### 1. Clone the repository

```bash
git clone https://github.com/annakhoriakova/property-value-prediction-neural-network.git
cd property-value-prediction-neural-network
```

### 2. Create a virtual environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install tensorflow pandas numpy
```

## Usage
Ensure the dataset file exists: Place yyc_dataset.csv in the same directory as main.py.

Run the main script:

```bash
python main.py
```
The script will:
1. Load and preprocess the dataset
2. Train all three models sequentially
3. Display training progress for each model
4. Show final test MAE (Mean Absolute Error) results

## Understanding the Results
**Mean Absolute Error (MAE):** Represents the average dollar difference between predicted and actual property values.

**Underfitting Model:** Will have high MAE on both training and test data due to insufficient capacity.

**Overfitting Model:** Will have very low MAE on training data but high MAE on test data due to memorization.

**Optimized Model:** Should achieve the best test MAE by balancing capacity and regularization.
