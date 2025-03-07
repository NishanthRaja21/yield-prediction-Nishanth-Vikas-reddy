import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Load dataset
st.set_page_config(page_title="Crop Yield Prediction Dashboard", layout="wide")
st.title("🌾 Crop Yield Prediction Using Random Forest")

# File Upload
uploaded_file = st.file_uploader("📂 Upload Crop Yield Dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded Successfully!")
else:
    st.warning("⚠️ Please upload the dataset to proceed.")
    st.stop()

# Sidebar Controls
st.sidebar.header("🔧 Model Configuration")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42, step=1)

# Display dataset preview
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Preprocessing
X = df.drop(columns=['Yield'])
y = df['Yield']
X = pd.get_dummies(X, columns=['Crop', 'Season', 'State'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model training
model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate performance metrics
# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Handle zero values in y_test for MAPE calculation
y_test_no_zeros = np.where(y_test == 0, np.nan, y_test)  # Replace zeros with NaN
mape = np.nanmean(np.abs((y_test - y_pred) / y_test_no_zeros)) * 100  # Ignore zero-division errors
accuracy = 100 - mape if not np.isnan(mape) else 0  # Ensure accuracy is a valid number

# Display Evaluation Metrics
st.subheader("Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy (100 - MAPE)", f"{accuracy:.2f}%")
col2.metric("📉 MAE", f"{mae:.4f}")
col3.metric("📈 R² Score", f"{r2:.4f}")

st.write(f"📊 **Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"📈 **Explained Variance Score:** {evs:.4f}")


# Plot Actual vs Predicted Yield
st.subheader("📊 Actual vs Predicted Yield")
fig, ax = plt.subplots()
sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={"alpha": 0.5})
ax.set_xlabel("Actual Yield")
ax.set_ylabel("Predicted Yield")
ax.set_title("Actual vs Predicted Yield")
st.pyplot(fig)

# User Prediction Input
st.sidebar.subheader("🎯 Predict Yield for Custom Input")
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.number_input(f"Enter {col}", value=float(X[col].mean()))

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])
user_prediction = model.predict(user_df)[0]

st.sidebar.success(f"🌱 Predicted Yield: {user_prediction:.2f}")

st.write("🔍 Adjust the sidebar inputs to see how they impact the prediction!")