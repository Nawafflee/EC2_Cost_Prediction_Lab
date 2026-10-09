# EC2 Instance Cost Prediction Dashboard

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Configure the Streamlit page
st.set_page_config(
    page_title="EC2 Cost Prediction Dashboard",
    page_icon="☁️",
    layout="wide"
)

# Dashboard title
st.title("Amazon EC2 Instance Cost Prediction")

# Dashboard introduction
st.write(
    "This dashboard uses Linear Regression to estimate "
    "Amazon EC2 On-Demand hourly costs based on "
    "instance memory and vCPU count."
)

st.info(
    "The application uses a regression model trained "
    "on historical EC2 pricing data. Predictions are "
    "estimates, not official AWS price quotations."
)


# ---------------------------------------------------------
# LOAD DATA AND TRAIN REGRESSION MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_ec2_model():

    # Load the original EC2 dataset
    data = pd.read_csv("ec2dataset.csv")

    # Convert On-Demand prices into numeric values
    data["On Demand"] = pd.to_numeric(
        data["On Demand"].str.replace(
            "[$, hourly]", "", regex=True
        ),
        errors="coerce"
    )

    # Convert memory and vCPUs into numeric values
    data["Instance Memory"] = pd.to_numeric(
        data["Instance Memory"].str.replace(" GiB", "")
    )

    data["vCPUs"] = pd.to_numeric(
        data["vCPUs"].str.extract(r"(\d+)", expand=False)
    )

    # Remove records missing required model values
    data_cleaned = data.dropna(
        subset=["On Demand", "Instance Memory", "vCPUs"]
    )

    # Define model features and target
    X = data_cleaned[["Instance Memory", "vCPUs"]]
    y = data_cleaned["On Demand"]

    # Reproduce the professor's 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train the Linear Regression model

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate predictions for the testing dataset
    y_pred = model.predict(X_test)

    # Return the model, dataset counts, and evaluation data
    return (
        model,
        len(data_cleaned),
        len(X_train),
        len(X_test),
        y_test,
        y_pred
    )



# Train and retrieve the model

(
    model,
    total_instances,
    training_count,
    testing_count,
    y_test,
    y_pred
) = train_ec2_model()



# ---------------------------------------------------------
# MODEL TRAINING SUMMARY
# ---------------------------------------------------------

st.subheader("Regression Model Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Valid EC2 Instances", total_instances)
col2.metric("Training Instances", training_count)
col3.metric("Testing Instances", testing_count)

st.success("Linear Regression model loaded and trained successfully.")


# ---------------------------------------------------------
# EC2 INSTANCE CONFIGURATION
# ---------------------------------------------------------

st.subheader("EC2 Configuration")

st.write(
    "Enter the memory and vCPU configuration "
    "for the EC2 instance you want to estimate."
)

# Arrange inputs side by side
col1, col2 = st.columns(2)

with col1:
    memory_gib = st.number_input(
        "Instance Memory (GiB)",
        min_value=0.5,
        max_value=32768.0,
        value=4.0,
        step=0.5
    )

with col2:
    vcpu_count = st.number_input(
        "Number of vCPUs",
        min_value=1,
        max_value=896,
        value=2,
        step=1
    )

# Display the selected configuration
st.write(
    f"Selected configuration: "
    f"{memory_gib} GiB memory and "
    f"{vcpu_count} vCPUs."
)


# ---------------------------------------------------------
# EC2 COST PREDICTION
# ---------------------------------------------------------

st.subheader("EC2 Cost Prediction")

# Generate a prediction when the user clicks the button
if st.button("Predict On-Demand Cost", type="primary"):

    # Prepare input data using the same feature names
    # that were used during model training
    new_instance = pd.DataFrame({
        "Instance Memory": [memory_gib],
        "vCPUs": [vcpu_count]
    })

    # Predict the hourly cost
    predicted_hourly_cost = float(
        model.predict(new_instance)[0]
    )

    # Check for unrealistic negative predictions
    if predicted_hourly_cost < 0:

        st.error(
            "The regression model produced a negative "
            "hourly estimate, which is not a valid EC2 price."
        )

        st.write(
            f"Raw model output: "
            f"${predicted_hourly_cost:.4f}/hour"
        )

        st.warning(
            "This configuration cannot receive a usable "
            "cost estimate from the current regression model."
        )

    else:

        # Estimate monthly cost using 730 hours per month
        predicted_monthly_cost = predicted_hourly_cost * 730

        # Display prediction metrics
        hourly_col, monthly_col = st.columns(2)

        hourly_col.metric(
            "Estimated Hourly Cost",
            f"${predicted_hourly_cost:,.4f}"
        )

        monthly_col.metric(
            "Estimated Monthly Cost",
            f"${predicted_monthly_cost:,.2f}"
        )

        st.caption(
            "Monthly estimates assume 730 operating hours. "
            "Predictions are approximate and do not include "
            "additional AWS service charges."
        )

    st.info(
        "Model estimates are based only on memory and vCPUs. "
        "Actual EC2 prices also depend on factors such as "
        "instance family, AWS Region, and operating system."
    )


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.subheader("Model Performance")

st.write(
    "The following metrics evaluate the Linear Regression "
    "model using EC2 instances reserved for testing."
)

# Calculate regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Display performance metrics
mae_col, mse_col, rmse_col = st.columns(3)

mae_col.metric(
    "Mean Absolute Error (MAE)",
    f"${mae:.4f}/hr"
)

mse_col.metric(
    "Mean Squared Error (MSE)",
    f"{mse:.4f}"
)

rmse_col.metric(
    "Root Mean Squared Error (RMSE)",
    f"${rmse:.4f}/hr"
)

st.caption(
    "Evaluation is based on the 20% testing dataset. "
    "Lower error values indicate better predictions. "
    "MSE is expressed in squared USD/hour units."
)


# ---------------------------------------------------------
# ACTUAL VS. PREDICTED COST VISUALIZATION
# ---------------------------------------------------------

st.subheader("Actual vs. Predicted EC2 Costs")

st.write(
    "This scatterplot compares actual EC2 On-Demand prices "
    "with the prices estimated by the Linear Regression model."
)

# Create the scatterplot
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    y_test,
    y_pred,
    color="blue",
    alpha=0.6,
    label="EC2 Test Instances"
)

# Reference line representing perfect predictions
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

# Configure chart labels
ax.set_xlabel("Actual On-Demand Cost (USD/hour)")
ax.set_ylabel("Predicted On-Demand Cost (USD/hour)")
ax.set_title("Actual vs. Predicted EC2 Instance Costs")

ax.legend()
fig.tight_layout()

# Display the chart inside Streamlit
st.pyplot(fig)

# Release the Matplotlib figure
plt.close(fig)

st.caption(
    "Points closer to the red dashed line represent more "
    "accurate predictions. Larger deviations indicate "
    "greater prediction errors."
)


