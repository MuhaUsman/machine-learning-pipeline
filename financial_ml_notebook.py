import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

print("Financial ML Pipeline - Sample Data Generator")
print("=" * 50)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Download sample stock data from Yahoo Finance
print("\nDownloading sample stock data from Yahoo Finance...")
ticker = "AAPL"
period = "1y"

try:
    aapl_data = yf.download(ticker, period=period, progress=False)
    print(f"Successfully downloaded {ticker} stock data ({len(aapl_data)} rows)")
    print("Preview:")
    print(aapl_data.head())
    
    # Save the data to a CSV file
    aapl_data.to_csv("sample_stock_data.csv")
    print(f"Saved {ticker} stock data to sample_stock_data.csv")
except Exception as e:
    print(f"Error downloading stock data: {str(e)}")

# 2. Create a synthetic financial dataset
print("\nCreating synthetic financial dataset...")

# Generate dates for one year
dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="B")
n_samples = len(dates)

# Create a synthetic financial dataset
synthetic_data = pd.DataFrame({
    'Date': dates,
    'Market_Index': np.cumsum(np.random.normal(0.001, 0.02, n_samples)),
    'Interest_Rate': np.random.normal(2.0, 0.5, n_samples) / 100,
    'Inflation_Rate': np.random.normal(3.5, 0.7, n_samples) / 100,
    'Unemployment_Rate': np.random.normal(5.0, 0.3, n_samples) / 100,
    'Consumer_Sentiment': np.random.normal(85, 5, n_samples),
    'Oil_Price': np.random.normal(80, 10, n_samples),
    'Exchange_Rate': np.random.normal(1.1, 0.05, n_samples),
})

# Add some noise and trends
synthetic_data['Market_Index'] = synthetic_data['Market_Index'] * 1000 + 3500
synthetic_data['Market_Index'] = synthetic_data['Market_Index'] * (1 + 0.2 * np.sin(np.arange(n_samples) / 50))

# Create target variable (Stock Price) that depends on other features
synthetic_data['Stock_Price'] = (
    100 + 
    20 * synthetic_data['Market_Index'] / 4000 + 
    -500 * synthetic_data['Interest_Rate'] + 
    -300 * synthetic_data['Inflation_Rate'] + 
    -100 * synthetic_data['Unemployment_Rate'] + 
    0.2 * synthetic_data['Consumer_Sentiment'] + 
    0.05 * synthetic_data['Oil_Price'] + 
    10 * synthetic_data['Exchange_Rate'] + 
    np.random.normal(0, 5, n_samples)  # Add some random noise
)

print("Preview of synthetic data:")
print(synthetic_data.head())

# Save the synthetic dataset to a CSV file
synthetic_data.to_csv("kragle_financial_data.csv", index=False)
print("Saved synthetic financial data to kragle_financial_data.csv")

print("\nSample data generation complete!")
print("\nTo run the Streamlit app, execute the following command in your terminal:")
print("streamlit run app.py") 