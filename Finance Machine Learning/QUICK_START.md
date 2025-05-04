# Financial ML Pipeline - Quick Start Guide

This guide will help you quickly get started with the Financial ML Pipeline application.

## Setup

1. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

2. **Generate sample data (optional):**
   ```
   python financial_ml_notebook.py
   ```
   This will create two sample datasets:
   - `sample_stock_data.csv`: Real AAPL stock data from Yahoo Finance
   - `kragle_financial_data.csv`: Synthetic financial dataset

3. **Run the application:**
   ```
   streamlit run app.py
   ```

4. **Open the application** in your browser at http://localhost:8501

## Using the Application

### 1. Load Data
- **Option A:** Upload a CSV/XLSX file (try `kragle_financial_data.csv`)
- **Option B:** Fetch stock data by entering a ticker symbol (e.g., "AAPL", "MSFT")

### 2. Preprocess Data
- Click the "Preprocess Data" button to clean the data
- Review statistics and visualizations of the cleaned data

### 3. Engineer Features
- Click the "Engineer Features" button to create new features
- Examine feature correlation matrix and importance

### 4. Split Data
- Click the "Split Data" button to divide data into training/testing sets
- View the split distribution

### 5. Train Model
- Click the "Train Model" button to train the selected model
- For K-Means, you can select the optimal number of clusters

### 6. Evaluate Model
- Click the "Evaluate Model" button to see performance metrics
- Review visualizations of model performance

### 7. Visualize Results
- Click the "Visualize Results" button for detailed visualizations
- Download predictions or cluster assignments

## Tips
- Use the sidebar to switch between different models
- For regression/classification, select an appropriate target column
- Download processed data at any stage for further analysis
- Each step must be completed in sequence

## Example Workflow
1. Load AAPL stock data
2. Preprocess to handle missing data and outliers
3. Engineer features to create technical indicators
4. Split into 80% training, 20% testing
5. Train a Linear Regression model
6. Evaluate performance using MSE, MAE, and R²
7. Visualize predictions vs actual values 