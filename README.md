# 🏠 California Housing Price Predictor (Streamlit App)

This project implements a **Multiple Linear Regression** model to predict median house values in California using the well-known California Housing Dataset. The application is built with **Streamlit**, providing an interactive and user-friendly interface for data exploration, model evaluation, and real-time price prediction.

## ✨ Features

*   **Interactive Prediction:** Predict house prices using a set of input features (Median Income, House Age, etc.) via a sidebar form.
*   **Model Evaluation:** Display key regression metrics (R², MSE, RMSE) and a plot comparing actual vs. predicted values.
*   **Data Exploration:** Visualize the relationship between Median Income and House Value with an interactive scatter plot.
*   **Feature Importance:** Show the model's coefficients to understand the influence of each feature on the prediction.
*   **Cached Performance:** Uses Streamlit's caching mechanism to ensure the data loading and model training steps run only once, providing fast performance.

## 🛠️ Technologies Used

*   **Python**
*   **Streamlit** - For building the web application.
*   **Scikit-learn** - For the Linear Regression model and dataset.
*   **Pandas & NumPy** - For data handling and numerical operations.
*   **Plotly Express** - For interactive data visualizations.

## 🚀 Getting Started

### Prerequisites

You need to have Python installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Amena-Akkad/california-housing-predictor.git
    cd california-housing-predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The application requires the following libraries. You can install them using `pip`:
    ```bash
    pip install streamlit scikit-learn pandas numpy plotly
    ```
    *(Note: `matplotlib` and `seaborn` are imported but not strictly necessary for the final `predictor.py` functionality, but are good to have for general data science environments.)*

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run predictor.py
    ```

2.  **Access the App:**
    The application will automatically open in your web browser at `http://localhost:8501`.

## 📁 Project Structure

The core of the project is a single file:

```
.
├── predictor.py          # The main Streamlit application code
└── README.md       # This file
```

## 💡 Code Highlights

The `predictor.py` file includes a key optimization using the `@st.cache_data` decorator:

```python
@st.cache_data
def load_data_and_train_model():
    """Loads the California Housing data and trains the Linear Regression model."""
    # ... data loading and model training logic ...
    return df, model, features, mse, rmse, r2, X_test, y_test
```

This ensures that the computationally expensive steps of loading the dataset and training the machine learning model are executed only once when the app starts, significantly improving the user experience on subsequent interactions.
