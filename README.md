# Laptop_Predict
Machine Learning Project with Python


Here’s a README for your project in English:

---

# Laptop Price Prediction

This project utilizes machine learning to predict the price of laptops based on their technical specifications. It includes steps for data cleaning, feature engineering, and building a regression model using `RandomForestRegressor`. The goal is to predict the price of new laptops based on the provided parameters.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Modifying Input Data](#modifying-input-data)
- [License](#license)

## Overview

The project reads a dataset containing laptop specifications and their prices. It preprocesses the data, selects relevant features, and trains a Random Forest regression model. Finally, it allows for predictions on new laptop configurations.

## Requirements

To run this project, you need the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preparation

1. **Load the dataset:**
   The dataset is loaded from a CSV file named `laptop_price.csv`.

2. **Data Cleaning:**
   The code cleans the dataset by:
   - Dropping non-essential columns.
   - Creating dummy variables for categorical features.
   - Extracting relevant information from string columns (e.g., CPU, RAM, GPU).
   - Converting string values to numerical types.

3. **Feature Selection:**
   The model selects features that correlate significantly with the target variable (laptop price).

## Model Training

The model is trained using the following steps:

1. **Split the Data:**
   The dataset is divided into training and testing sets (85% for training and 15% for testing).

2. **Standardization:**
   Features are standardized using `StandardScaler`.

3. **Train the Model:**
   A `RandomForestRegressor` is fitted to the training data.

4. **Hyperparameter Tuning:**
   `GridSearchCV` is used to find the best hyperparameters for the model.

## Making Predictions

To predict the price of a new laptop, use the following code snippet:

```python
# Example of new input data to predict the price of a laptop (replace with actual values)
new_data = pd.DataFrame({
    'Netbook': [0], 
    'No OS': [1],
    'Chrome OS': [0],
    'Ultrabook': [0],
    'AMD_Gpu': [1],
    'Weight': [1.34],
    'Windows 10': [1],
    'Linux': [0],
    'Intel_Gpu': [1],
    'Workstation': [0],
    'Acer': [0],
    'Razer': [1],
    'MSI': [0],
    'Nvidia_Gpu': [1],
    'Gaming': [0],
    'CPU Frequency': [2.5],
    'Screen Height': [2160],
    'Notebook': [1],
    'Screen Width': [1920],
    'Ram': [16],
})

new_data_scaled = scaler.transform(new_data)

# Use the trained model to predict the price
predicted_price = best_forest.predict(new_data_scaled)

print(f"The predicted price of a new laptop is: € {predicted_price[0]:.2f}")
```

### Modifying Input Data

You can modify the `new_data` DataFrame to predict the price of different laptop configurations. Here’s a brief explanation of each column:

- **Netbook:** Binary value indicating if the laptop is a netbook.
- **No OS:** Binary value indicating if the laptop has no operating system.
- **Chrome OS:** Binary value indicating if the laptop runs Chrome OS.
- **Ultrabook:** Binary value indicating if the laptop is an ultrabook.
- **AMD_Gpu, Intel_Gpu, Nvidia_Gpu:** Binary values indicating the presence of respective GPU brands.
- **Weight:** The weight of the laptop in kilograms.
- **Windows 10, Linux, Workstation:** Binary values indicating the operating system type.
- **Screen Height & Width:** The screen dimensions in pixels.
- **CPU Frequency:** The CPU frequency in GHz.
- **Ram:** The amount of RAM in GB.

Feel free to adjust these values based on the laptop specifications you want to analyze.

Feel free to modify any sections to better fit your project or to add additional information as needed!
