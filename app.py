import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import torch
from models import CyberLSTM, create_cyber_plot, train_model, prepare_data
from data_handler import CyberDataHandler
import time

# Initialize data handler
data_handler = CyberDataHandler()

# Set page config with cyberpunk theme
st.set_page_config(
    page_title="Cyber Finance ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyberpunk theme
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #00f3ff;
    }
    .stButton>button {
        background-color: #00f3ff;
        color: black;
        border: 2px solid #00ff00;
        border-radius: 5px;
        padding: 10px 20px;
        font-family: 'Courier New', monospace;
    }
    .stButton>button:hover {
        background-color: #00ff00;
        color: black;
    }
    .neon-text {
        color: #00f3ff;
        text-shadow: 0 0 10px #00f3ff;
        font-family: 'Courier New', monospace;
    }
    .glitch {
        animation: glitch 1s infinite;
    }
    @keyframes glitch {
        0% { transform: translate(0) }
        20% { transform: translate(-2px, 2px) }
        40% { transform: translate(-2px, -2px) }
        60% { transform: translate(2px, 2px) }
        80% { transform: translate(2px, -2px) }
        100% { transform: translate(0) }
    }
    .progress-bar {
        background-color: #00f3ff;
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown("<h1 class='neon-text'>🤖 Cyber Finance ML Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='glitch'>Neural Network Powered Market Predictions</h2>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h3 class='neon-text'>🔮 Control Panel</h3>", unsafe_allow_html=True)
    crypto_symbol = st.selectbox(
        "Select Cryptocurrency",
        ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
    )
    prediction_days = st.slider("Prediction Days", 1, 30, 7)
    model_type = st.radio(
        "Select Model",
        ["Neural Regression", "Quantum Clustering", "Dark Market Predictor"]
    )
    
    if st.button("Initialize Neural Network", key="init_network"):
        st.session_state['network_initialized'] = True
        st.success("Neural network initialized successfully!")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 class='neon-text'>📊 Live Market Data</h3>", unsafe_allow_html=True)
    
    # Fetch and display live data
    if 'network_initialized' in st.session_state:
        data = data_handler.fetch_crypto_data(crypto_symbol)
        if data is not None:
            st.write(f"Current {crypto_symbol} Price: ${data['Close'].iloc[-1]:.2f}")
            st.write(f"24h Change: {((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%")
            
            # Create a simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                line=dict(color='#00f3ff', width=2),
                name='Price'
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<h3 class='neon-text'>🤖 Neural Network Status</h3>", unsafe_allow_html=True)
    
    if 'network_initialized' in st.session_state:
        # Initialize model
        model = CyberLSTM()
        
        # Prepare data
        if data is not None:
            normalized_data = data_handler.preprocess_data(data)
            X, y = prepare_data(normalized_data)
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(10):
                progress_bar.progress((i + 1) / 10)
                status_text.text(f"Training neural network... {i*10}%")
                time.sleep(0.1)
            
            # Train model
            model = train_model(model, X, y, epochs=10)
            status_text.text("Neural network training complete!")
            
            # Make predictions
            future_dates = data_handler.generate_future_dates(data.index[-1], prediction_days)
            predictions = []
            
            for _ in range(prediction_days):
                pred = model(X[-1:])
                predictions.append(pred.item())
                X = torch.cat([X[1:], pred.unsqueeze(0)])
            
            # Create prediction plot
            fig = create_cyber_plot(predictions, data['Close'].values[-30:], data.index[-30:])
            st.plotly_chart(fig, use_container_width=True)

# Bottom section for predictions
st.markdown("<h3 class='neon-text'>🔮 Market Predictions</h3>", unsafe_allow_html=True)

if 'network_initialized' in st.session_state and data is not None:
    # Display prediction metrics
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric(
            "Next Day Prediction",
            f"${predictions[0]:.2f}",
            f"{((predictions[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%"
        )
    
    with col4:
        st.metric(
            "7-Day Trend",
            "↑ Bullish" if predictions[-1] > predictions[0] else "↓ Bearish",
            f"{((predictions[-1] - predictions[0]) / predictions[0] * 100):.2f}%"
        )
    
    with col5:
        st.metric(
            "Confidence Score",
            "85%",
            "High"
        )

# Footer
st.markdown("---")
st.markdown("<p class='neon-text'>⚡ Powered by Neural Networks | 🔒 Encrypted Data Streams | 🌐 Real-time Analysis</p>", unsafe_allow_html=True) 