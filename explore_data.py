import pandas as pd

# Load our EC2 dataset.
data = pd.read_csv("ec2dataset.csv")

# Display the first five rows.
print("First five EC2 instances:")
print(data.head())

# Display the number of rows and columns.
print("\nDataset dimensions:")
print(data.shape)

# Display column names, data types, and missing-value information.
print("\nDataset information:")
data.info()


# List of cost-related columns
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

# Remove '$' and 'hourly' and convert columns to numeric
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Check for missing values after conversion
print(data[cost_columns].isnull().sum())


# Step 4: Perform Summary Analysis

# Generate summary statistics for cost-related columns
cost_summary = data[cost_columns].describe()

print(cost_summary)


# Step 5: Visualize the Data

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
sns.set(style="whitegrid")

# Create a boxplot to visualize the distribution of costs
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2")

# Set the plot labels and title
plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

# Adjust the plot layout
plt.tight_layout()

# Step 6: Identify Outliers

# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Find outliers in the On-Demand cost column
outliers_on_demand = detect_outliers('On Demand')

# Display the detected outliers
print("\nOn-Demand Cost Outliers:")
print(outliers_on_demand)

# Display the total number of outliers
print("\nTotal On-Demand Outliers:", len(outliers_on_demand))

# Display the plot after the analysis is complete
plt.show()


# Step 7: Conclusion and Further Analysis

# Compare Reserved and On-Demand costs
cost_comparison = data[
    ['Name', 'On Demand', 'Linux Reserved cost']
].dropna().sort_values('On Demand')

# Display the 10 lowest-cost instances
print("\nTop 10 Lowest-Cost EC2 Instances:")
print(cost_comparison.head(10))


# Step 8: Filter and Compare Different Instance Families

# Filter for specific instance families
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]

# Example: Filter for T2 and T3 instance families
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

# Compare summary statistics for T2 and T3 instances
t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("T2 Instance Costs Summary:\n", t2_summary)
print("\nT3 Instance Costs Summary:\n", t3_summary)


# Step 8.2: Visualize T2 Instance Cost Distribution

# Visualize cost comparison for T2 instances
plt.figure(figsize=(12, 6))

# Add T2 instances
sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True
)

plt.title('Cost Distribution for T2 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.tight_layout()

# Display all prepared figures
plt.show()


# Step 8.2B: Visualize T3 Instance Cost Distribution

# Create a new figure for T3 instances
plt.figure(figsize=(12, 6))

# Add T3 instances
sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True
)

plt.title('Cost Distribution for T3 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.tight_layout()


# Step 8.3: Compare On-Demand and Reserved Costs
# for T2 and T3 Instance Families

# Compare On-Demand and Reserved costs for T2 and T3 families
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])

# Sort by On-Demand costs
comparison_sorted = comparison.dropna().sort_values('On Demand')

# Display the 10 lowest-cost instances
print("\nT2 and T3 On-Demand vs Reserved Comparison:")
print(comparison_sorted.head(10))


# Display all prepared figures
plt.show()

