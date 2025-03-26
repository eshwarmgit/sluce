import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Generate Sample Data
def generate_data(n=300):
    random.seed(42)
    surface_area = np.random.uniform(1, 500, n)  # in km²
    depth = np.random.uniform(5, 50, n)  # in meters
    breadth = np.random.uniform(1, 100, n)  # in km
    previous_maintenance_cost = np.random.uniform(50000, 500000, n)
    market_rate = np.random.uniform(1.1, 1.5, n)
    baseline_cost = surface_area * depth * breadth * 0.3  # Hypothetical base cost
    auction_price = baseline_cost * market_rate + np.random.uniform(-10000, 10000, n)
    
    # Simulating data for last 5 years
    depth_prev = depth + np.random.uniform(-5, 5, n)
    breadth_prev = breadth + np.random.uniform(-5, 5, n)
    surface_area_prev = surface_area + np.random.uniform(-10, 10, n)
    
    # Determining lake condition and maintenance suggestions
    condition = []
    maintenance_suggestions = []
    maintenance_frequency = []
    for d, dp, b, bp, sa, sap in zip(depth, depth_prev, breadth, breadth_prev, surface_area, surface_area_prev):
        if d < dp:
            condition.append("High Sedimentation")
            maintenance_suggestions.append("Increase depth by dredging and use sediment for shore strengthening.")
            maintenance_frequency.append("High (Every Year)")
        elif b > bp:
            condition.append("High Erosion")
            maintenance_suggestions.append("Strengthen shores to prevent erosion.")
            maintenance_frequency.append("Moderate (Every 2-3 Years)")
        else:
            condition.append("Stable")
            maintenance_suggestions.append("Regular maintenance of inlets and outlets required.")
            maintenance_frequency.append("Low (Every 5 Years)")
    
    df = pd.DataFrame({
        'Surface_Area': surface_area,
        'Depth': depth,
        'Breadth': breadth,                                                                                                                                                                                                                         
        'Condition': condition,
        'Maintenance_Suggestions': maintenance_suggestions,
        'Maintenance_Frequency': maintenance_frequency,
        'Prev_Maintenance_Cost': previous_maintenance_cost,
        'Auction_Price': auction_price
    })
    return df

data = generate_data(300)
data.to_csv("lake_data.csv", index=False)

# Load Data
df = pd.read_csv("lake_data.csv")

# Preprocessing
encoder = OneHotEncoder()
categorical_cols = ["Condition"]
categorical_data = encoder.fit_transform(df[categorical_cols]).toarray()
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out())
df = df.drop(columns=categorical_cols).join(categorical_df)

# Split Data
X = df.drop(columns=["Auction_Price", "Maintenance_Suggestions", "Maintenance_Frequency"])
y = df["Auction_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Deployment with Streamlit
st.title("Irrigation Lake Maintenance Predictor")
st.sidebar.header("Enter Lake Details")
surface_area = st.sidebar.slider("Surface Area (km²)", 1.0, 500.0, 50.0)
depth = st.sidebar.slider("Depth (m)", 5.0, 50.0, 20.0)
breadth = st.sidebar.slider("Breadth (km)", 1.0, 100.0, 10.0)
prev_cost = st.sidebar.number_input("Previous Maintenance Cost", 50000, 500000, 100000)

depth_prev = depth + np.random.uniform(-5, 5)
breadth_prev = breadth + np.random.uniform(-5, 5)
surface_area_prev = surface_area + np.random.uniform(-10, 10)

# Automate Condition Calculation and Maintenance Suggestion
if depth < depth_prev:
    condition = "High Sedimentation"
    maintenance_suggestion = "Increase depth by dredging and use sediment for shore strengthening."
    maintenance_frequency = "High (Every Year)"
elif breadth > breadth_prev:
    condition = "High Erosion"
    maintenance_suggestion = "Strengthen shores to prevent erosion."
    maintenance_frequency = "Moderate (Every 2-3 Years)"
else:
    condition = "Stable"
    maintenance_suggestion = "Regular maintenance of inlets and outlets required."
    maintenance_frequency = "Low (Every 5 Years)"

st.write(f"### Computed Lake Condition: {condition}")
st.write(f"### Suggested Maintenance: {maintenance_suggestion}")
st.write(f"### Maintenance Frequency: {maintenance_frequency}")

input_data = pd.DataFrame([[surface_area, depth, breadth, prev_cost, condition]], columns=["Surface_Area", "Depth", "Breadth", "Prev_Maintenance_Cost", "Condition"])
input_data = input_data.join(pd.DataFrame(encoder.transform(input_data[["Condition"]]).toarray(), columns=encoder.get_feature_names_out()))
input_data = input_data.drop(columns=["Condition"])

if st.sidebar.button("Predict Maintenance Cost"):
    pred_price = model.predict(input_data)[0]
    st.write(f"Predicted Maintenance Cost: ₹{pred_price:,.2f}")
