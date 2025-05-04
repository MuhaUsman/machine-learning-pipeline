import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import base64
from PIL import Image
import io
import os
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Financial ML Pipeline",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def add_custom_styling():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&display=swap');
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
    }
    
    body {
        font-family: 'Quicksand', sans-serif !important;
        background-color: #f6f1d1;
    }
    
    .stButton > button {
        background-color: #5E8271 !important;
        color: white !important;
        border: 2px solid #f6f1d1 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #f6f1d1 !important;
        color: #5E8271 !important;
        border: 2px solid #5E8271 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #e9e4c4;
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%235E8271' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    
    .main .block-container {
        background-color: #ffffff;
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%235E8271' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        padding: 2rem;
        border: 3px double #5E8271;
        border-radius: 10px;
    }
    
    .vintage-border {
        border: 8px solid #fff;
        border-image: url('data:image/svg+xml;utf8,<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M0,0 L100,0 L100,100 L0,100 L0,0 Z M10,10 L10,90 L90,90 L90,10 L10,10 Z" fill="none" stroke="%235E8271" stroke-width="2"/><path d="M10,10 L25,10 M75,10 L90,10 M10,90 L25,90 M75,90 L90,90 M10,10 L10,25 M10,75 L10,90 M90,10 L90,25 M90,75 L90,90" stroke="%23f6f1d1" stroke-width="3"/></svg>') 20 stretch;
        padding: 20px;
        margin: 20px 0;
    }
    
    .success-animation {
        background-color: rgba(94, 130, 113, 0.1);
        border-left: 5px solid #5E8271;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_styling()

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Helper functions
def load_welcome_gif():
    """Display a finance-themed welcome GIF"""
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="https://static.wixstatic.com/media/00a0e0_698052cbe3fa44fca4dbba7f52ab293f~mv2.gif" width="500px" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    """, unsafe_allow_html=True)

def vintage_border(content_function):
    """Wrapper to add vintage border styling to a section"""
    st.markdown('<div class="vintage-border">', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)

def success_animation(message):
    """Display a success message with animation"""
    st.markdown(f"""
    <div class="success-animation">
        <div style="display: flex; align-items: center;">
            <div style="margin-right: 15px;">
                <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHFxeGszenRicDM2MXp6Z3BxNnMzeHJhaDg2NDl3NHFqMmNtMHF1YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/XbgzkRGjS0EX3LvQ7a/giphy.gif" width="40px">
            </div>
            <div>
                <h4 style="margin: 0; font-family: 'Quicksand', sans-serif;">{message}</h4>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #5E8271; text-decoration: none; border: 1px solid #5E8271; padding: 5px 10px; border-radius: 3px; background-color: #f8f9fa;"><i class="fas fa-download"></i> {text}</a>'
    return href

def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance with improved error handling"""
    try:
        print(f"Attempting to fetch {ticker} data for period {period}...")
        stock = yf.Ticker(ticker)
        
        # First try the standard history method
        data = stock.history(period=period)
        
        # Check if data is empty
        if data.empty:
            # Try with explicit start and end dates as fallback
            end_date = pd.Timestamp.now()
            if period == "1mo":
                start_date = end_date - pd.Timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - pd.Timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - pd.Timedelta(days=180)
            elif period == "1y":
                start_date = end_date - pd.Timedelta(days=365)
            elif period == "2y":
                start_date = end_date - pd.Timedelta(days=730)
            elif period == "5y":
                start_date = end_date - pd.Timedelta(days=1825)
            else:
                start_date = end_date - pd.Timedelta(days=365)
                
            print(f"Trying with explicit dates: {start_date} to {end_date}")
            data = stock.history(start=start_date, end=end_date)
        
        # If we still have no data, use a direct download as last resort
        if data.empty:
            print("Using direct download as fallback...")
            data = yf.download(ticker, period=period, progress=False)
        
        # Final check for empty data
        if data.empty:
            return None, f"No data found for {ticker}. The symbol may not exist or there might be API restrictions."
            
        print(f"Successfully fetched {len(data)} rows of {ticker} data")
        return data, None
    except Exception as e:
        error_msg = str(e)
        print(f"Error fetching {ticker} data: {error_msg}")
        
        # If it's a connection issue, suggest checking internet connection
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return None, f"Connection error while fetching {ticker} data. Please check your internet connection and try again."
        
        return None, f"Error fetching {ticker} data: {error_msg}"

def preprocess_data(data):
    """Preprocess the data - clean NAs, remove outliers"""
    # Create a copy to avoid modifying the original data
    processed_df = data.copy()
    
    # Handle missing values
    processed_df = processed_df.dropna()
    
    # Handle outliers (using IQR method)
    if len(processed_df) > 20:  # Only if we have enough data points
        # Get only numeric columns
        numeric_cols = processed_df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:  # Only proceed if we have numeric columns
            # Calculate IQR for numeric columns only
            Q1 = processed_df[numeric_cols].quantile(0.25)
            Q3 = processed_df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            
            # Identify outliers in numeric columns
            outlier_mask = ((processed_df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                           (processed_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            
            # Remove outliers
            processed_df = processed_df[~outlier_mask]
    
    return processed_df

def engineer_features(data, target_column=None, model_type=None):
    """Engineer features for the model"""
    df = data.copy()
    X = None
    y = None
    
    # For stock data
    if 'Open' in df.columns and 'Close' in df.columns:
        # Add technical indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['5D_MA'] = df['Close'].rolling(window=5).mean()
        df['20D_MA'] = df['Close'].rolling(window=20).mean()
        df['5D_Std'] = df['Close'].rolling(window=5).std()
        df['Upper_Band'] = df['5D_MA'] + (df['5D_Std'] * 2)
        df['Lower_Band'] = df['5D_MA'] - (df['5D_Std'] * 2)
        df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
        
        # Drop NAs created by rolling windows
        df = df.dropna()
        
        if model_type == 'Linear Regression':
            # Target: Next day closing price
            df['Next_Close'] = df['Close'].shift(-1)
            df = df.dropna()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
                       '5D_MA', '20D_MA', '5D_Std', 'Upper_Band', 'Lower_Band', 'MA_Ratio']
            X = df[features]
            y = df['Next_Close']
            
        elif model_type == 'Logistic Regression':
            # Target: Price goes up (1) or down (0) next day
            df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
                       '5D_MA', '20D_MA', '5D_Std', 'Upper_Band', 'Lower_Band', 'MA_Ratio']
            X = df[features]
            y = df['Price_Up']
            
        elif model_type == 'K-Means Clustering':
            # No target for clustering, just features
            features = ['Daily_Return', '5D_Std', 'MA_Ratio']
            X = df[features]
            y = None
    
    # For custom datasets with a specified target column
    elif target_column and target_column in df.columns:
        # Drop non-numeric columns
        df = df.select_dtypes(include=['number'])
        
        # For classification, convert target to binary if needed
        if model_type == 'Logistic Regression':
            # If target is not binary, convert based on median
            if df[target_column].nunique() > 2:
                median = df[target_column].median()
                df['target_binary'] = (df[target_column] > median).astype(int)
                features = [col for col in df.columns if col != target_column and col != 'target_binary']
                X = df[features]
                y = df['target_binary']
            else:
                features = [col for col in df.columns if col != target_column]
                X = df[features]
                y = df[target_column]
        else:
            features = [col for col in df.columns if col != target_column]
            X = df[features]
            y = df[target_column]
    
    # General case - no specific structure known
    else:
        # Drop non-numeric columns
        df = df.select_dtypes(include=['number'])
        
        if model_type == 'K-Means Clustering':
            X = df
            y = None
        elif len(df.columns) > 1:
            # Use the last column as target by default
            target_col = df.columns[-1]
            features = [col for col in df.columns if col != target_col]
            X = df[features]
            y = df[target_col]
            
            # For logistic regression, convert target to binary if needed
            if model_type == 'Logistic Regression' and y.nunique() > 2:
                median = y.median()
                y = (y > median).astype(int)
    
    return df, X, y 

# Main application UI
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #5E8271;'>Financial ML Pipeline</h1>", unsafe_allow_html=True)
    
    # Welcome section - Removing the box around this section
    load_welcome_gif()
    st.markdown("<h2 style='text-align: center; font-family: \"Quicksand\", sans-serif;'>Welcome to the Finance Machine Learning Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center;'>
    This application guides you through building a Machine Learning pipeline for financial data analysis.
    Upload your own dataset or fetch real-time stock data to get started.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar - Data Input
    with st.sidebar:
        st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Data Sources</h3>", unsafe_allow_html=True)
        
        # File uploader
        st.markdown("<p style='font-family: \"Quicksand\", sans-serif;'>Upload Kragle Dataset:</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"], label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Yahoo Finance data
        st.markdown("<p style='font-family: \"Quicksand\", sans-serif;'>Fetch Yahoo Finance Data:</p>", unsafe_allow_html=True)
        ticker_symbol = st.text_input("Ticker Symbol", "AAPL")
        time_period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"])
        
        fetch_data_button = st.button("Fetch Stock Data")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Model selection
        st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Model Selection</h3>", unsafe_allow_html=True)
        model_type = st.radio("Choose a Model", 
                             ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                             key="model_selection")
        st.session_state.model_type = model_type
        
        if st.session_state.data is not None:
            if model_type != "K-Means Clustering":
                numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                if numeric_columns:
                    target_column = st.selectbox("Target Column (for Regression/Classification)", 
                                               numeric_columns,
                                               index=len(numeric_columns)-1 if len(numeric_columns) > 0 else 0)
                    st.session_state.target_column = target_column
    
    # Load Data Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 1: Load Data</h3>", unsafe_allow_html=True)
    
    # Load data from uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.session_state.data_source = "upload"
            
            st.write("Data Preview:")
            st.write(data.head())
            
            success_animation("Kragle dataset loaded successfully!")
            
            st.markdown(get_table_download_link(data), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Load data from Yahoo Finance
    if fetch_data_button:
        with st.spinner('Fetching stock data...'):
            data, error = fetch_stock_data(ticker_symbol, period=time_period)
            
            if error:
                st.error(f"Error fetching stock data: {error}")
                
                # Suggest using sample data if Yahoo Finance fails
                if st.button("Use Sample Stock Data Instead"):
                    try:
                        # Check if we have sample data from the data generator
                        sample_file = "sample_stock_data.csv"
                        if os.path.exists(sample_file):
                            data = pd.read_csv(sample_file, index_col=0, parse_dates=True)
                            st.session_state.data = data
                            st.session_state.data_source = "sample"
                        else:
                            # Create synthetic stock data as last resort
                            st.info("Creating synthetic stock data for demonstration purposes...")
                            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='B')
                            price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)) * 0.5)
                            data = pd.DataFrame({
                                'Open': price * (1 - np.random.random(len(dates)) * 0.01),
                                'High': price * (1 + np.random.random(len(dates)) * 0.01),
                                'Low': price * (1 - np.random.random(len(dates)) * 0.01),
                                'Close': price,
                                'Volume': np.random.randint(1000000, 10000000, len(dates))
                            }, index=dates)
                            st.session_state.data = data
                            st.session_state.data_source = "synthetic"
                        
                        st.write("Sample Stock Data Preview:")
                        st.write(data.head())
                        
                        # Plot the stock price
                        fig = px.line(data, y='Close', title=f'Sample Stock Price')
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            font=dict(family="Quicksand"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        success_animation("Sample stock data loaded successfully!")
                        
                        st.markdown(get_table_download_link(data, filename="sample_stock_data.csv"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
            else:
                st.session_state.data = data
                st.session_state.data_source = "yfinance"
                
                st.write("Stock Data Preview:")
                st.write(data.head())
                
                # Plot the stock price
                fig = px.line(data, y='Close', title=f'{ticker_symbol} Stock Price')
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    font=dict(family="Quicksand"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                success_animation("Stock data fetched successfully!")
                
                st.markdown(get_table_download_link(data, filename=f"{ticker_symbol}_data.csv"), unsafe_allow_html=True)
    
    # Preprocessing Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 2: Preprocessing</h3>", unsafe_allow_html=True)
    
    preprocess_button = st.button("Preprocess Data")
    
    if preprocess_button and st.session_state.data is not None:
        with st.spinner('Preprocessing data...'):
            processed_data = preprocess_data(st.session_state.data)
            st.session_state.preprocessed_data = processed_data
            
            # Show stats
            st.write("Data statistics after preprocessing:")
            st.write(processed_data.describe())
            
            # Show removed rows count
            original_rows = len(st.session_state.data)
            processed_rows = len(processed_data)
            removed_rows = original_rows - processed_rows
            
            if removed_rows > 0:
                st.info(f"Removed {removed_rows} rows with missing values or outliers.")
            
            # Before vs After comparison chart
            if 'Close' in processed_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.data.index,
                    y=st.session_state.data['Close'],
                    mode='lines',
                    name='Original Data',
                    line=dict(color='#f6f1d1')
                ))
                fig.add_trace(go.Scatter(
                    x=processed_data.index,
                    y=processed_data['Close'],
                    mode='lines',
                    name='Processed Data',
                    line=dict(color='#5E8271')
                ))
                fig.update_layout(
                    title="Original vs. Preprocessed Data",
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    font=dict(family="Quicksand"),
                    legend=dict(x=0, y=1, traceorder="normal"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            success_animation("Data preprocessing complete!")
            
            st.markdown(get_table_download_link(processed_data, filename="preprocessed_data.csv", 
                                             text="Download Preprocessed Data"), unsafe_allow_html=True)
    
    # Feature Engineering Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 3: Feature Engineering</h3>", unsafe_allow_html=True)
    
    feature_engineering_button = st.button("Engineer Features")
    
    if feature_engineering_button and st.session_state.preprocessed_data is not None:
        with st.spinner('Engineering features...'):
            target_column = st.session_state.target_column if 'target_column' in st.session_state else None
            
            engineered_df, X, y = engineer_features(
                st.session_state.preprocessed_data,
                target_column=target_column,
                model_type=st.session_state.model_type
            )
            
            st.session_state.engineered_df = engineered_df
            st.session_state.X = X
            st.session_state.y = y
            
            st.write("Engineered Features Preview:")
            st.write(engineered_df.head())
            
            # Display feature importance/correlation
            if X is not None and len(X.columns) > 0:
                st.write("Feature Correlation Matrix:")
                corr = X.corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=["#f6f1d1", "#FFFFFF", "#5E8271"],
                    aspect="auto"
                )
                fig.update_layout(
                    font=dict(family="Quicksand"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # If we have a target, show correlation with target
                if y is not None and st.session_state.model_type != "K-Means Clustering":
                    # Create a DataFrame with correlations to target
                    if isinstance(y, pd.Series):
                        target_corr = pd.DataFrame({
                            'Feature': X.columns,
                            'Correlation': [X[col].corr(y) for col in X.columns]
                        }).sort_values('Correlation', ascending=False)
                        
                        # Plot feature importance bar chart
                        fig = px.bar(
                            target_corr,
                            x='Feature',
                            y='Correlation',
                            title=f"Feature Correlation with {target_column}",
                            color='Correlation',
                            color_continuous_scale=["#f6f1d1", "#FFFFFF", "#5E8271"],
                        )
                        fig.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Correlation",
                            font=dict(family="Quicksand"),
                            xaxis={'categoryorder':'total descending'},
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            success_animation("Feature engineering complete!")
            
            st.markdown(get_table_download_link(engineered_df, filename="engineered_data.csv", 
                                             text="Download Engineered Data"), unsafe_allow_html=True)
    
    # Train/Test Split Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 4: Train/Test Split</h3>", unsafe_allow_html=True)
    
    split_button = st.button("Split Data")
    
    if split_button and 'X' in st.session_state and st.session_state.X is not None:
        with st.spinner('Splitting data...'):
            # For supervised learning, split data into train/test sets
            if st.session_state.model_type != "K-Means Clustering" and st.session_state.y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    st.session_state.X, st.session_state.y, test_size=0.2, random_state=42
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Display split information
                st.write(f"Training set: {X_train.shape[0]} samples")
                st.write(f"Testing set: {X_test.shape[0]} samples")
                
                # Pie chart showing the split
                fig = px.pie(
                    values=[X_train.shape[0], X_test.shape[0]],
                    names=['Training Set', 'Testing Set'],
                    title='Train/Test Split',
                    color_discrete_sequence=['#5E8271', '#f6f1d1']
                )
                fig.update_layout(
                    font=dict(family="Quicksand"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                success_animation("Data split complete!")
            
            # For unsupervised learning (K-Means)
            else:
                st.session_state.X_train = st.session_state.X
                st.session_state.X_test = None
                st.session_state.y_train = None
                st.session_state.y_test = None
                
                st.write(f"Data ready for clustering: {st.session_state.X.shape[0]} samples")
                success_animation("Data prepared for unsupervised learning!")
    
    # Model Training Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 5: Model Training</h3>", unsafe_allow_html=True)
    
    train_button = st.button("Train Model")
    
    if train_button and 'X_train' in st.session_state and st.session_state.X_train is not None:
        with st.spinner('Training model...'):
            # Scale features (especially important for K-Means)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            st.session_state.scaler = scaler
            
            if st.session_state.X_test is not None:
                X_test_scaled = scaler.transform(st.session_state.X_test)
                st.session_state.X_test_scaled = X_test_scaled
            
            st.session_state.X_train_scaled = X_train_scaled
            
            # Train the selected model
            if st.session_state.model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train_scaled, st.session_state.y_train)
                
            elif st.session_state.model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42)
                model.fit(X_train_scaled, st.session_state.y_train)
                
            elif st.session_state.model_type == "K-Means Clustering":
                # Determine optimal number of clusters using Elbow Method
                st.write("Determining optimal number of clusters...")
                
                inertia = []
                k_range = range(1, min(10, st.session_state.X_train.shape[0]))
                for k in k_range:
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_model.fit(X_train_scaled)
                    inertia.append(kmeans_model.inertia_)
                
                # Plot Elbow Method
                elbow_df = pd.DataFrame({
                    'Number of Clusters': list(k_range),
                    'Inertia': inertia
                })
                
                fig = px.line(
                    elbow_df, 
                    x='Number of Clusters', 
                    y='Inertia',
                    title='Elbow Method for Optimal K',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Inertia",
                    font=dict(family="Quicksand"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Default to 3 clusters or let user choose
                optimal_k = st.slider("Select number of clusters:", min_value=2, max_value=min(9, st.session_state.X_train.shape[0]), value=3)
                
                model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                model.fit(X_train_scaled)
            
            st.session_state.model = model
            
            success_animation("Model training complete!")
    
    # Model Evaluation Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 6: Model Evaluation</h3>", unsafe_allow_html=True)
    
    evaluate_button = st.button("Evaluate Model")
    
    if evaluate_button and 'model' in st.session_state and st.session_state.model is not None:
        with st.spinner('Evaluating model...'):
            # For supervised learning models
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                # Make predictions
                y_pred = st.session_state.model.predict(st.session_state.X_test_scaled)
                st.session_state.predictions = y_pred
                
                # Evaluate based on model type
                if st.session_state.model_type == "Linear Regression":
                    mse = mean_squared_error(st.session_state.y_test, y_pred)
                    mae = mean_absolute_error(st.session_state.y_test, y_pred)
                    r2 = r2_score(st.session_state.y_test, y_pred)
                    
                    st.write("Model Evaluation Metrics:")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R² Score'],
                        'Value': [mse, mae, r2]
                    })
                    st.table(metrics_df)
                    
                    # Plot actual vs predicted values
                    results_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': y_pred
                    }).reset_index(drop=True)
                    
                    fig = px.scatter(
                        results_df, 
                        x='Actual', 
                        y='Predicted',
                        trendline='ols',
                        title='Actual vs Predicted Values'
                    )
                    fig.update_layout(
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values",
                        font=dict(family="Quicksand"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    # Add a perfect prediction line
                    x_range = [min(results_df['Actual']), max(results_df['Actual'])]
                    fig.add_trace(go.Scatter(
                        x=x_range, 
                        y=x_range,
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='#f6f1d1', dash='dash')
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    success_animation("Linear Regression evaluation complete!")
                    
                elif st.session_state.model_type == "Logistic Regression":
                    accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    
                    st.write("Model Evaluation Metrics:")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy'],
                        'Value': [accuracy]
                    })
                    st.table(metrics_df)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    cm_df = pd.DataFrame(
                        cm, 
                        index=['Actual 0', 'Actual 1'], 
                        columns=['Predicted 0', 'Predicted 1']
                    )
                    
                    fig = px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale=["#FFFFFF", "#5E8271"],
                        title='Confusion Matrix',
                        aspect="auto"
                    )
                    fig.update_layout(
                        font=dict(family="Quicksand"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    success_animation("Logistic Regression evaluation complete!")
            
            # For unsupervised learning (K-Means)
            elif st.session_state.model_type == "K-Means Clustering":
                # Get cluster assignments
                cluster_labels = st.session_state.model.labels_
                st.session_state.cluster_labels = cluster_labels
                
                # Add cluster labels to the original data
                clustered_data = st.session_state.X_train.copy()
                clustered_data['Cluster'] = cluster_labels
                
                st.write("Data with Cluster Assignments:")
                st.write(clustered_data.head())
                
                # Visualize clusters (selecting 2 features for visualization)
                if st.session_state.X_train.shape[1] >= 2:
                    # Select the first two features for visualization
                    features = st.session_state.X_train.columns[:2]
                    
                    # Create a DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'Feature1': st.session_state.X_train[features[0]],
                        'Feature2': st.session_state.X_train[features[1]],
                        'Cluster': cluster_labels
                    })
                    
                    fig = px.scatter(
                        plot_df, 
                        x='Feature1', 
                        y='Feature2',
                        color='Cluster',
                        color_continuous_scale=["#5E8271", "#f6f1d1"],
                        title=f'Cluster Visualization using {features[0]} and {features[1]}'
                    )
                    fig.update_layout(
                        xaxis_title=features[0],
                        yaxis_title=features[1],
                        font=dict(family="Quicksand"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add cluster centers
                    if st.session_state.X_train.shape[1] >= 2:
                        # Transform cluster centers back to original scale
                        centers = st.session_state.model.cluster_centers_
                        centers_original = st.session_state.scaler.inverse_transform(centers)
                        
                        # Create a DataFrame with cluster centers
                        centers_df = pd.DataFrame(
                            centers_original[:, :2],
                            columns=[features[0], features[1]]
                        )
                        centers_df['Cluster'] = range(len(centers))
                        
                        st.write("Cluster Centers:")
                        st.write(centers_df)
                
                success_animation("K-Means Clustering analysis complete!")
    
    # Results Visualization Step
    st.markdown("<h3 style='font-family: \"Quicksand\", sans-serif; color: #5E8271;'>Step 7: Results Visualization</h3>", unsafe_allow_html=True)
    
    visualize_button = st.button("Visualize Results")
    
    if visualize_button and 'model' in st.session_state and st.session_state.model is not None:
        with st.spinner('Generating visualizations...'):
            # For regression models
            if st.session_state.model_type == "Linear Regression" and 'predictions' in st.session_state:
                # Create a time-series plot for stock data
                if 'Close' in st.session_state.engineered_df.columns:
                    # Get the dates from the original test data
                    test_indices = st.session_state.X_test.index
                    
                    # Create a DataFrame with actual and predicted values
                    results_df = pd.DataFrame({
                        'Date': test_indices,
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.predictions
                    })
                    
                    # Create an interactive time-series plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Date'],
                        y=results_df['Actual'],
                        mode='lines',
                        name='Actual Price',
                        line=dict(color='#5E8271')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['Date'],
                        y=results_df['Predicted'],
                        mode='lines',
                        name='Predicted Price',
                        line=dict(color='#f6f1d1')
                    ))
                    fig.update_layout(
                        title='Actual vs Predicted Stock Prices',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        font=dict(family="Quicksand"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (from coefficients)
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': np.abs(st.session_state.model.coef_)
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale=["#5E8271", "#f6f1d1"]
                )
                fig.update_layout(
                    xaxis_title='Feature',
                    yaxis_title='Absolute Coefficient Value',
                    font=dict(family="Quicksand"),
                    xaxis={'categoryorder':'total descending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a download link for the predictions
                pred_df = pd.DataFrame({
                    'Actual': st.session_state.y_test,
                    'Predicted': st.session_state.predictions,
                    'Error': st.session_state.y_test - st.session_state.predictions
                })
                st.markdown(get_table_download_link(pred_df, filename="predictions.csv", 
                                                 text="Download Predictions"), unsafe_allow_html=True)
                
                success_animation("Regression results visualization complete!")
            
            # For classification models
            elif st.session_state.model_type == "Logistic Regression" and 'predictions' in st.session_state:
                # Create a download link for the predictions
                pred_df = pd.DataFrame({
                    'Actual': st.session_state.y_test,
                    'Predicted': st.session_state.predictions,
                    'Correct': st.session_state.y_test == st.session_state.predictions
                })
                st.markdown(get_table_download_link(pred_df, filename="predictions.csv", 
                                                 text="Download Predictions"), unsafe_allow_html=True)
                
                # Feature importance (from coefficients)
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': np.abs(st.session_state.model.coef_[0])
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale=["#5E8271", "#f6f1d1"]
                )
                fig.update_layout(
                    xaxis_title='Feature',
                    yaxis_title='Absolute Coefficient Value',
                    font=dict(family="Quicksand"),
                    xaxis={'categoryorder':'total descending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Accuracy by feature value (for most important feature)
                if len(feature_importance) > 0:
                    top_feature = feature_importance.iloc[0]['Feature']
                    
                    # Get the feature values from test data
                    feature_values = st.session_state.X_test[top_feature]
                    
                    # Create bins for the feature values
                    bins = pd.qcut(feature_values, q=5, duplicates='drop')
                    
                    # Calculate accuracy for each bin
                    bin_accuracy = []
                    bin_labels = []
                    
                    for bin_name, group in pred_df.groupby(bins):
                        accuracy = (group['Actual'] == group['Predicted']).mean()
                        bin_accuracy.append(accuracy)
                        bin_labels.append(f"{bin_name.left:.2f} to {bin_name.right:.2f}")
                    
                    # Create a DataFrame for plotting
                    bin_df = pd.DataFrame({
                        'Bin': bin_labels,
                        'Accuracy': bin_accuracy
                    })
                    
                    fig = px.bar(
                        bin_df,
                        x='Bin',
                        y='Accuracy',
                        title=f'Accuracy by {top_feature} Value',
                        color='Accuracy',
                        color_continuous_scale=["#f6f1d1", "#5E8271"]
                    )
                    fig.update_layout(
                        xaxis_title=f'{top_feature} Range',
                        yaxis_title='Accuracy',
                        font=dict(family="Quicksand"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                success_animation("Classification results visualization complete!")
            
            # For clustering models
            elif st.session_state.model_type == "K-Means Clustering" and 'cluster_labels' in st.session_state:
                # Cluster distribution
                cluster_counts = pd.Series(st.session_state.cluster_labels).value_counts().sort_index()
                
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    title='Sample Distribution by Cluster',
                    labels={'x': 'Cluster', 'y': 'Number of Samples'},
                    color=cluster_counts.values,
                    color_continuous_scale=["#5E8271", "#f6f1d1"]
                )
                fig.update_layout(
                    xaxis_title='Cluster',
                    yaxis_title='Number of Samples',
                    font=dict(family="Quicksand"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D Visualization if we have at least 3 features
                if st.session_state.X_train.shape[1] >= 3:
                    features = st.session_state.X_train.columns[:3]
                    
                    plot_df = pd.DataFrame({
                        'Feature1': st.session_state.X_train[features[0]],
                        'Feature2': st.session_state.X_train[features[1]],
                        'Feature3': st.session_state.X_train[features[2]],
                        'Cluster': st.session_state.cluster_labels
                    })
                    
                    fig = px.scatter_3d(
                        plot_df,
                        x='Feature1',
                        y='Feature2',
                        z='Feature3',
                        color='Cluster',
                        title=f'3D Cluster Visualization',
                        color_continuous_scale=["#5E8271", "#59BC91", "#f6f1d1"],
                        opacity=0.7
                    )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title=features[0],
                            yaxis_title=features[1],
                            zaxis_title=features[2]
                        ),
                        font=dict(family="Quicksand"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a download link for the cluster assignments
                clustered_data = st.session_state.X_train.copy()
                clustered_data['Cluster'] = st.session_state.cluster_labels
                st.markdown(get_table_download_link(clustered_data, filename="cluster_assignments.csv", 
                                                 text="Download Cluster Assignments"), unsafe_allow_html=True)
                
                success_animation("Clustering results visualization complete!")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd;">
        <p style="color: #5E8271; font-family: 'Quicksand', sans-serif;">
            Financial ML Pipeline | Powered by Streamlit & Python
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 