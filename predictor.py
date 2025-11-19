import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# --- Configuration and Setup ---
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Model Training (Cached) ---
@st.cache_data
def load_data_and_train_model():
    """Loads the California Housing data and trains the Linear Regression model."""
    # Load data
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Select features
    features = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'Latitude', 'Longitude']
    X = df[features]
    y = df['MedHouseVal']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # IMPORTANT FIX: Return X_test and y_test for use in the main script
    return df, model, features, mse, rmse, r2, X_test, y_test

df, model, features, mse, rmse, r2, X_test, y_test = load_data_and_train_model()

# --- Streamlit Interface ---
st.title("🏠 California Housing Price Predictor")
st.markdown("""
A **Multiple Linear Regression** model built on the **California Housing Dataset** to predict median house values.
The model uses demographic and geographic factors to provide an estimate.
""")

# --- Sidebar for Prediction ---
st.sidebar.header("🔮 Predict New House Price")
with st.sidebar.form("prediction_form"):
    st.subheader("Input Features")
    med_inc = st.slider("Median Income (in $10k):", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("House Age:", 1, 55, 20)
    ave_rooms = st.slider("Average Number of Rooms:", 1.0, 15.0, 5.0, 0.1)
    population = st.slider("Population:", 100, 35000, 1000, 100)
    latitude = st.slider("Latitude:", 32.0, 42.0, 36.0, 0.1)
    longitude = st.slider("Longitude:", -124.0, -114.0, -120.0, 0.1)
    
    predict_button = st.form_submit_button("Predict Price")

if predict_button:
    input_data = np.array([[med_inc, house_age, ave_rooms, population, latitude, longitude]])
    predicted_price = model.predict(input_data)[0]
    
    # The target variable is in hundreds of thousands of dollars ($100,000)
    st.sidebar.success(f"💰 Predicted House Value: **${predicted_price * 100000:,.2f}**")
    st.balloons()

# --- Main Content Area ---
tab1, tab2, tab3 = st.tabs(["Model Overview", "Data Exploration", "Feature Importance"])

with tab1:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("R-squared (R²)", f"{r2:.4f}", help="Proportion of the variance in the dependent variable that is predictable from the independent variables.")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}", help="Average squared difference between the estimated values and the actual value.")
    col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}", help="Square root of the MSE, providing error in the same units as the target variable.")
    
    st.subheader("Prediction vs. Actual Value Plot")
    # Create a simple plot of predicted vs actual values for the test set
    y_test_array = y_test.values
    y_pred_array = model.predict(X_test)
    
    plot_df = pd.DataFrame({
        'Actual Value': y_test_array,
        'Predicted Value': y_pred_array
    })
    
    fig_scatter = px.scatter(plot_df, x='Actual Value', y='Predicted Value', 
                             title='Actual vs. Predicted House Values (Test Set)',
                             labels={'Actual Value': 'Actual House Value ($100k)', 'Predicted Value': 'Predicted House Value ($100k)'})
    fig_scatter.add_shape(
        type='line', line=dict(dash='dash', color='red'),
        x0=plot_df['Actual Value'].min(), y0=plot_df['Actual Value'].min(),
        x1=plot_df['Actual Value'].max(), y1=plot_df['Actual Value'].max()
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.header("Data Exploration")
    st.subheader("Data Preview")
    num_rows = st.slider("Number of rows to display:", min_value=5, max_value=50, value=10, step=5)
    st.dataframe(df.head(num_rows))
    
    st.subheader("Relationship Between Income and House Price")
    threshold = st.slider("Select Y-axis reference threshold (House Value in $100k):", 0.5, 5.0, 2.0, 0.1)
    
    fig_income = px.scatter(df, x='MedInc', y='MedHouseVal', color='HouseAge',
                         labels={'MedInc': 'Median Income ($10k)', 'MedHouseVal': 'House Value ($100k)'},
                         title='Effect of Income on House Price by Property Age',
                         hover_data=['Latitude', 'Longitude'])
    
    fig_income.add_shape(
        type='line',
        x0=df['MedInc'].min(), x1=df['MedInc'].max(),
        y0=threshold, y1=threshold,
        line=dict(color='red', dash='dash'),
    )
    
    fig_income.add_annotation(
        x=df['MedInc'].mean(),
        y=threshold + 0.1,
        text=f"Reference Threshold = ${threshold * 100000:,.0f}",
        showarrow=False,
        font=dict(color="red")
    )
    
    st.plotly_chart(fig_income, use_container_width=True)

with tab3:
    st.header("Feature Importance")
    st.markdown("The coefficients from the Linear Regression model indicate the weight and direction of each feature's influence on the predicted house value.")
    
    coeff_df = pd.DataFrame({
        'Feature': features, 
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    st.dataframe(coeff_df, use_container_width=True)
    
    # Visualize coefficients
    fig_coeff = px.bar(coeff_df, x='Coefficient', y='Feature', orientation='h',
                       title='Model Coefficients (Feature Importance)',
                       color='Coefficient',
                       color_continuous_scale=px.colors.diverging.RdBu)
    st.plotly_chart(fig_coeff, use_container_width=True)

st.markdown("---")
st.caption("Data Source: California Housing Dataset (scikit-learn)")
