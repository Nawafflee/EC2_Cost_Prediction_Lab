
# Part 2: Predicting EC2 Instance Costs Using Regression Analysis

# Step 2: Load and Clean the Data

import pandas as pd

# Load the dataset
file_path = 'ec2dataset.csv'
data = pd.read_csv(file_path)

# Clean the cost columns
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Verify the data was loaded and cleaned
print("Dataset dimensions:", data.shape)

print("\nMissing pricing values:")
print(data[cost_columns].isnull().sum())


# Step 3: Feature Engineering

# Convert memory from string (e.g., '0.5 GiB') to numeric
data['Instance Memory'] = pd.to_numeric(
    data['Instance Memory'].str.replace(' GiB', '')
)

# Extract the number of vCPUs (e.g., '2 vCPUs')
data['vCPUs'] = pd.to_numeric(
    data['vCPUs'].str.extract(r'(\d+)', expand=False)
)

# Check the data after conversion
print("\nMemory and vCPU data after conversion:")
print(data[['Instance Memory', 'vCPUs']].head())


# Step 4: Handle Missing Data

# Drop rows with missing values in the relevant columns
data_cleaned = data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)

# Verify that the cleaned data has no missing values
print("\nMissing Values After Cleaning:")
print(data_cleaned.isnull().sum())

# Display the number of remaining instances
print("\nCleaned Dataset Dimensions:")
print(data_cleaned.shape)


# Step 5: Split the Data

from sklearn.model_selection import train_test_split

# Define features and target
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']

# Split the data into training and testing sets
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# Step 6: Train a Linear Regression Model

from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Print the coefficients of the model
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")


# Step 7: Evaluate the Model

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict the costs for the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# Step 8: Visualize Actual vs. Predicted Costs

import matplotlib.pyplot as plt

# Plot actual costs against predicted costs
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    color='blue',
    alpha=0.6
)

# Add a reference line for perfect predictions
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--'
)

plt.xlabel('Actual On-Demand Cost (USD)')
plt.ylabel('Predicted On-Demand Cost (USD)')
plt.title('Actual vs. Predicted EC2 Instance Costs')

plt.tight_layout()
plt.show()


# Step 9: Predict the Cost of a New EC2 Instance

# Define a new instance with 4 GiB memory and 2 vCPUs
new_instance = pd.DataFrame({
    'Instance Memory': [4],
    'vCPUs': [2]
})

# Predict the hourly On-Demand cost
predicted_cost = model.predict(new_instance)

# Display the prediction
print("\nNew EC2 Instance Cost Prediction:")
print("Memory: 4 GiB")
print("vCPUs: 2")
print(f"Predicted On-Demand Cost: ${predicted_cost[0]:.4f}/hour")

# Display the regression visualization
plt.show()
