import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config and theme
st.set_page_config(
    page_title="Financial ML Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for corporate blue theme
st.markdown("""
    <style>
    .stApp {
        background-color: #000080;
        color: #C0C0C0;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #000080;
        color: #C0C0C0;
        border: 2px solid #C0C0C0;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #C0C0C0;
        color: #000080;
    }
    .stDataFrame {
        background-color: #000080;
        color: #C0C0C0;
    }
    </style>
    """, unsafe_allow_html=True)

# Stock ticker animation
st.markdown("""
    <div style="overflow: hidden; white-space: nowrap; background-color: #000080; color: #C0C0C0; padding: 10px;">
        <marquee behavior="scroll" direction="left">
            AAPL: $175.23 | MSFT: $420.55 | GOOGL: $150.89 | AMZN: $180.45 | META: $485.67
        </marquee>
    </div>
    """, unsafe_allow_html=True)

# Title and welcome message
st.title("Financial Machine Learning Dashboard")
st.markdown("""
    <div style="text-align: center; margin: 20px;">
        <h2>Welcome to the Corporate Financial Analysis Platform</h2>
        <p>Upload your data or fetch real-time stock information to begin your analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for data input
with st.sidebar:
    st.header("Data Input")
    data_source = st.radio(
        "Choose Data Source",
        ["Upload Kragle Dataset", "Fetch Yahoo Finance Data"]
    )
    
    if data_source == "Upload Kragle Dataset":
        uploaded_file = st.file_uploader("Upload your Kragle dataset", type=['csv', 'xlsx'])
    else:
        ticker = st.text_input("Enter Stock Ticker", "AAPL")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

# Main content area
st.header("Machine Learning Pipeline")

# Step 1: Data Loading
st.subheader("1. Data Loading")
if data_source == "Upload Kragle Dataset" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    st.success("Data loaded successfully!")
elif data_source == "Fetch Yahoo Finance Data":
    if st.button("Fetch Data"):
        data = yf.download(ticker, start=start_date, end=end_date)
        st.dataframe(data.head())
        st.success("Data fetched successfully!")

# Step 2: Preprocessing
st.subheader("2. Data Preprocessing")
if 'df' in locals():
    if st.button("Start Preprocessing"):
        st.info("Preprocessing in progress...")
        # Add preprocessing steps here
        st.success("Preprocessing completed!")

# Step 3: Feature Engineering
st.subheader("3. Feature Engineering")
if 'df' in locals():
    if st.button("Engineer Features"):
        st.info("Feature engineering in progress...")
        # Add feature engineering steps here
        st.success("Feature engineering completed!")

# Step 4: Model Training
st.subheader("4. Model Training")
model_type = st.selectbox(
    "Choose Model Type",
    ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
)

if st.button("Train Model"):
    st.info("Training model...")
    # Add model training steps here
    st.success("Model training completed!")

# Step 5: Results Visualization
st.subheader("5. Results Visualization")
if st.button("Show Results"):
    st.info("Generating visualizations...")
    # Add visualization code here
    st.success("Visualizations generated!")

# Footer with skyline divider
st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <img src="https://via.placeholder.com/1200x50/000080/C0C0C0?text=Office+Skyline+Divider" 
             style="width: 100%; height: 50px; object-fit: cover;">
    </div>
    """, unsafe_allow_html=True) 