# Financial Volatility Prediction System

This repository contains my bachelor's thesis project - a comprehensive financial volatility prediction system that combines statistical methods, machine learning, and deep learning techniques to predict and analyze stock and index volatility.

![Financial Volatility Prediction](https://img.shields.io/badge/Financial-Volatility%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 🌟 Features

- **Multi-Model Approach**: Combines statistical methods (GARCH), machine learning (Random Forest), and deep learning (LSTM, Transformer) models
- **Comprehensive Analysis**: Processes financial data with various technical indicators and volatility metrics
- **Multiple Volatility Windows**: Supports different prediction windows (7, 14, 21, 30 days)
- **Model Optimization**: Optional hyperparameter optimization using Optuna
- **Performance Comparison**: Evaluates and compares models using multiple metrics (RMSE, R², MAE, MSE)
- **Visualization**: Generates comprehensive data analysis and model comparison visualizations
- **Web Interface**: User-friendly web application for running predictions and viewing results
- **Historical Results**: Saves prediction history for easy reference

## 🔧 Technologies Used

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, YFinance
- **Statistical Models**: Arch (GARCH implementation)
- **Machine Learning**: Scikit-learn (Random Forest)
- **Deep Learning**: PyTorch (LSTM and Transformer models)
- **Hyperparameter Optimization**: Optuna
- **Visualization**: Matplotlib
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5

## 📋 Project Structure

```
Financial-Volatility-Prediction/
├── app.py                # Flask web application
├── success.py            # Core prediction engine
├── templates/
│   └── index.html        # Web interface template
├── output/               # Output directory (created at runtime)
│   ├── data/             # Processed data
│   ├── models/           # Trained models
│   ├── predictions/      # Prediction results
│   ├── evaluations/      # Model evaluation metrics
│   └── visualizations/   # Generated charts and visualizations
└── README.md             # This file
```

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/financial-volatility-prediction.git
   cd financial-volatility-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the web application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## 📦 Dependencies

Create a `requirements.txt` with the following packages:

```
pandas
numpy
matplotlib
yfinance
torch
scikit-learn
statsmodels
arch
optuna
joblib
flask
```

## 📊 Model Overview

The system implements and compares four different approaches to volatility prediction:

1. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**
   - Statistical model specifically designed for volatility forecasting
   - Captures volatility clustering in financial time series

2. **Random Forest**
   - Ensemble machine learning method
   - Provides feature importance analysis for better model interpretability

3. **LSTM (Long Short-Term Memory)**
   - Deep learning approach for sequential data
   - Captures long-term dependencies in time series

4. **Transformer**
   - Attention-based deep learning architecture
   - State-of-the-art performance for many sequence modeling tasks

## 🎯 Usage

### Web Interface

1. Enter the stock/index symbol (e.g., `^GSPC` for S&P 500, `AAPL` for Apple Inc.)
2. Select the volatility window (7, 14, 21, or 30 days)
3. Choose the date range for analysis
4. Optionally enable hyperparameter optimization (takes longer but may improve results)
5. Click "Run Prediction" to start the analysis process
6. View the results in the visualization tabs
7. Previous predictions are saved in the history panel for easy reference

### Direct Code Usage

You can also use the prediction engine directly in your Python code:

```python
from success import run_volatility_prediction

results = run_volatility_prediction(
    symbol='^GSPC',                # S&P 500 index
    target_volatility_window=21,   # 21-day volatility window
    start_date='2018-01-01',
    end_date='2023-01-01',
    test_size=0.2,                 # 20% of data for testing
    seq_length=20,                 # Sequence length for deep learning models
    output_dir='output',
    optimize_params=False          # Set to True for hyperparameter optimization
)

# Access results
evaluations = results['evaluations']
target_col = results['target_col']
feature_cols = results['feature_cols']
```

## 📈 Evaluation Metrics

The system evaluates models using four key metrics:

- **RMSE (Root Mean Square Error)**: Measures the average magnitude of prediction errors
- **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in the data
- **MAE (Mean Absolute Error)**: Average absolute differences between predictions and actual values
- **MSE (Mean Squared Error)**: Average of squared differences between predictions and actual values

Lower values are better for RMSE, MAE, and MSE, while higher values (closer to 1) are better for R².

## 🔮 Future Improvements

Potential enhancements for this project:

1. Add more advanced deep learning architectures (e.g., Temporal Fusion Transformer)
2. Implement ensemble methods combining multiple models
3. Include more technical indicators and feature engineering techniques
4. Add support for cryptocurrency and forex data
5. Implement real-time prediction using live market data
6. Add explainable AI techniques for better model interpretability
7. Improve the frontend with interactive visualizations using D3.js or Plotly
8. Add multi-language support for the interface



*This project was created as part of a bachelor's thesis in Financial Engineering/Computer Science*
