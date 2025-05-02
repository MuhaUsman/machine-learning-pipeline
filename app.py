import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.inspection import permutation_importance
import joblib
import os
import time
import base64
import sys

# Add session state initialization at the top level
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'feature_scaler' not in st.session_state:
    st.session_state.feature_scaler = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Page configuration with improved metadata
st.set_page_config(
    page_title="ML Pipeline App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/streamlit_ml_app/issues',
        'Report a bug': 'https://github.com/yourusername/streamlit_ml_app/issues/new',
        'About': '''
        # ML Pipeline App
        An interactive machine learning pipeline application.
        Version: 1.0.0
        '''
    }
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme customization */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .Widget>label {
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.toast("The uploaded CSV file is empty!", icon="⚠️")
            return None
        st.toast("CSV file successfully loaded!", icon="✅")
        st.session_state.data = df  # Store in session state
        return df
    except pd.errors.EmptyDataError:
        st.toast("The uploaded file is empty!", icon="❌")
        return None
    except pd.errors.ParserError:
        st.toast("Error parsing the CSV file. Please check the file format.", icon="❌")
        return None
    except Exception as e:
        st.toast(f"An error occurred: {str(e)}", icon="❌")
        return None

# Add cache decorator for performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def validate_ticker(ticker):
    """Validate ticker symbol with enhanced error handling."""
    try:
        # Clean the ticker symbol
        ticker = ticker.strip().upper()
        
        # Basic format validation
        if not ticker or not ticker.isalnum():
            return False, "Invalid ticker format. Please use only letters and numbers."
        
        # Try to fetch ticker info
        stock = yf.Ticker(ticker)
        
        try:
            # First try to get basic info
            info = stock.info
        except (ValueError, Exception) as e:
            # Handle JSON decode error specifically
            if "Expecting value" in str(e):
                return False, f"Unable to validate ticker '{ticker}'. The symbol may not exist."
            return False, f"Error fetching data for '{ticker}': {str(e)}"
        
        # Verify we got valid data back
        if not info or len(info) == 0:
            return False, f"No data available for ticker '{ticker}'"
            
        if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
            return False, f"No current market data available for '{ticker}'"
            
        # Additional validation for common issues
        if 'symbol' in info and info['symbol'] != ticker:
            return False, f"Ticker mismatch: requested '{ticker}' but got '{info['symbol']}'"
            
        return True, f"Valid ticker symbol: {ticker}"
        
    except Exception as e:
        return False, f"Unexpected error validating '{ticker}': {str(e)}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with enhanced error handling and caching."""
    try:
        # Clean and validate the ticker
        ticker = ticker.strip().upper()
        
        # First validate the ticker
        is_valid, message = validate_ticker(ticker)
        if not is_valid:
            st.error(message)
            return None
            
        current_time = datetime.now()
        
        # Check if we need to refresh the data (60 minutes interval)
        if (st.session_state.last_refresh is None or 
            (current_time - st.session_state.last_refresh).seconds >= 3600):
            
            # Validate date range
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return None
                
            if end_date > current_time.date():
                st.warning("End date adjusted to today")
                end_date = current_time.date()
                
            # Add progress indicator
            with st.spinner(f"Fetching data for {ticker}..."):
                try:
                    df = yf.download(ticker, 
                                   start=start_date, 
                                   end=end_date, 
                                   progress=False,
                                   show_errors=False)
                    
                    if df.empty:
                        st.error(f"No data available for {ticker} in the specified date range")
                        return None
                        
                    # Verify we got all expected columns
                    expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                    if not all(col in df.columns for col in expected_columns):
                        st.error(f"Incomplete data received for {ticker}. Missing some price columns.")
                        return None
                    
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")
                    return None
            
            # Verify data quality
            if df.isnull().values.any():
                st.warning("Some data points are missing. Consider preprocessing.")
            
            st.session_state.last_refresh = current_time
            df.attrs['last_refresh'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            df.attrs['ticker'] = ticker
            
            # Add basic statistics
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)
            
            st.success(f"Successfully loaded data for {ticker}")
            st.session_state.data = df
            
            # Display basic info
            st.write("### Data Overview")
            st.write(f"Period: {start_date} to {end_date}")
            st.write(f"Trading days: {len(df)}")
            st.write(f"Current price: ${df['Close'].iloc[-1]:.2f}")
            st.write(f"Volume: {df['Volume'].iloc[-1]:,.0f}")
            
            return df
            
        else:
            # Use cached data
            df = st.session_state.data
            if df.attrs.get('ticker') != ticker:
                # If ticker changed, force refresh
                st.session_state.last_refresh = None
                return load_stock_data(ticker, start_date, end_date)
                
            time_since_refresh = current_time - datetime.strptime(
                df.attrs['last_refresh'], 
                '%Y-%m-%d %H:%M:%S'
            )
            
            st.info(
                f"Using cached data for {ticker}. "
                f"Last refresh: {df.attrs['last_refresh']} "
                f"({int(time_since_refresh.seconds/60)} minutes ago)"
            )
            return df
            
    except Exception as e:
        error_msg = str(e)
        if "HTTP 401" in error_msg:
            st.error("Authentication error. Please try again later.")
        elif "HTTP 429" in error_msg:
            st.error("Too many requests. Please wait a few minutes.")
        elif "HTTP 500" in error_msg:
            st.error("Server error. Please try again later.")
        elif "Connection" in error_msg:
            st.error("Connection error. Please check your internet connection.")
        else:
            st.error(f"An error occurred: {error_msg}")
        return None

def plot_missing_values(df):
    """Create a bar chart of missing values per column."""
    missing_values = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': (missing_values.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=True)
    
    if not missing_df.empty:
        fig = px.bar(
            missing_df,
            y='Column',
            x='Missing Values',
            text='Percentage',
            orientation='h',
            title='Missing Values Analysis',
            labels={'Column': 'Features', 'Missing Values': 'Number of Missing Values'},
            template='plotly_dark'
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        return fig
    return None

def validate_dataset(df, selected_features=None, target=None):
    """
    Validate dataset for model training.
    Returns: (is_valid, message, cleaned_df)
    """
    if df is None:
        return False, "No dataset provided", None
        
    # Create a copy to avoid modifying original data
    df_validated = df.copy()
    
    # Check for NaN values
    total_nan = df_validated.isna().sum().sum()
    if total_nan > 0:
        st.warning(f"Found {total_nan} missing values in the dataset")
        
    # If specific columns are provided, check them
    if selected_features is not None and target is not None:
        columns_to_check = selected_features + [target]
        df_validated = df_validated[columns_to_check]
    
    # Check for infinite values
    inf_mask = np.isinf(df_validated.select_dtypes(include=np.number))
    if inf_mask.values.any():
        st.warning("Found infinite values in the dataset. These will be replaced with NaN.")
        df_validated = df_validated.replace([np.inf, -np.inf], np.nan)
    
    # Check data types
    non_numeric = df_validated.select_dtypes(exclude=np.number).columns
    if len(non_numeric) > 0:
        return False, f"Non-numeric columns found: {', '.join(non_numeric)}", None
    
    # Check for constant columns
    constant_cols = [col for col in df_validated.columns if df_validated[col].nunique() == 1]
    if constant_cols:
        st.warning(f"Found constant columns that may affect model training: {', '.join(constant_cols)}")
    
    # Check for highly correlated features
    if len(df_validated.columns) > 1:
        corr_matrix = df_validated.corr().abs()
        high_corr = np.where(np.triu(corr_matrix, 1) > 0.95)
        high_corr_pairs = list(zip(corr_matrix.index[high_corr[0]], 
                                 corr_matrix.columns[high_corr[1]]))
        if high_corr_pairs:
            st.warning("Found highly correlated features (>0.95):")
            for feat1, feat2 in high_corr_pairs:
                st.write(f"- {feat1} & {feat2}: {corr_matrix.loc[feat1, feat2]:.2f}")
    
    return True, "Dataset validated successfully", df_validated

def handle_missing_values(df, strategies):
    """Apply missing value imputation strategies to the DataFrame with validation."""
    if df is None:
        return None
        
    df_processed = df.copy()
    
    # Store imputation statistics for reporting
    imputation_stats = {}
    
    for column, strategy in strategies.items():
        if strategy == 'drop':
            continue
            
        # Count missing values before imputation
        missing_before = df_processed[column].isna().sum()
        if missing_before == 0:
            continue
            
        # Store original statistics
        imputation_stats[column] = {
            'missing_before': missing_before,
            'strategy': strategy
        }
        
        try:
            if df[column].dtype in [np.number, float, int]:
                if strategy == 'mean':
                    value = df_processed[column].mean()
                    df_processed[column].fillna(value, inplace=True)
                elif strategy == 'median':
                    value = df_processed[column].median()
                    df_processed[column].fillna(value, inplace=True)
                elif strategy == 'zero':
                    df_processed[column].fillna(0, inplace=True)
                    value = 0
                
                # Store imputation value
                imputation_stats[column]['imputed_with'] = value
                
            else:
                if strategy == 'mode':
                    value = df_processed[column].mode()[0]
                    df_processed[column].fillna(value, inplace=True)
                    imputation_stats[column]['imputed_with'] = value
                elif strategy == 'empty':
                    df_processed[column].fillna('', inplace=True)
                    imputation_stats[column]['imputed_with'] = ''
        
        except Exception as e:
            st.error(f"Error during imputation of column {column}: {str(e)}")
            return None
    
    # Handle drops after imputation
    columns_to_drop = [col for col, strategy in strategies.items() if strategy == 'drop']
    if columns_to_drop:
        df_processed.drop(columns=columns_to_drop, inplace=True)
        for col in columns_to_drop:
            imputation_stats[col] = {'strategy': 'drop'}
    
    # Verify no missing values remain
    remaining_missing = df_processed.isna().sum()
    if remaining_missing.any():
        st.error("Some missing values remain after imputation:")
        for col, count in remaining_missing[remaining_missing > 0].items():
            st.write(f"- {col}: {count} missing values")
        return None
    
    # Display imputation summary
    st.write("### Imputation Summary")
    for column, stats in imputation_stats.items():
        if stats['strategy'] == 'drop':
            st.write(f"- {column}: Dropped column")
        else:
            st.write(f"- {column}: {stats['missing_before']} values imputed using {stats['strategy']}")
            if 'imputed_with' in stats:
                st.write(f"  Imputed with value: {stats['imputed_with']:.2f}" 
                        if isinstance(stats['imputed_with'], (float, int)) 
                        else f"  Imputed with value: {stats['imputed_with']}")
    
    return df_processed

def train_model(X, y, model_type, test_size=0.2, random_state=42):
    """Train the selected model with enhanced validation."""
    # Validate input data
    is_valid, message, X_validated = validate_dataset(X)
    if not is_valid:
        st.error(f"Feature validation failed: {message}")
        return None, None, None
        
    is_valid, message, y_validated = validate_dataset(pd.DataFrame(y))
    if not is_valid:
        st.error(f"Target validation failed: {message}")
        return None, None, None
    
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_validated, y_validated, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and train the model
        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'features': list(X.columns)
            }
        
        elif model_type == "Logistic Regression":
            # Verify target is binary
            unique_values = np.unique(y)
            if len(unique_values) != 2:
                st.error(f"Logistic Regression requires binary target. Found {len(unique_values)} classes.")
                return None, None, None
                
            model = LogisticRegression(random_state=random_state)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'test_size': len(y_test),
                'correct_predictions': (y_test == y_pred).sum(),
                'features': list(X.columns)
            }
        
        else:  # K-Means
            n_clusters = len(np.unique(y)) if y is not None else 3
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
            model.fit(X_train_scaled)
            metrics = {
                'inertia': model.inertia_,
                'silhouette': silhouette_score(X_train_scaled, model.labels_),
                'features': list(X.columns)
            }
        
        return model, metrics, scaler
        
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.error("Debugging information:")
        st.write("- Feature shape:", X.shape)
        st.write("- Target shape:", y.shape if y is not None else "N/A")
        st.write("- Feature NaN count:", X.isna().sum().sum())
        st.write("- Target NaN count:", y.isna().sum() if y is not None else "N/A")
        return None, None, None

def create_prediction_plot(y_true, y_pred, model_type):
    """Create an interactive scatter plot comparing actual vs predicted values."""
    if model_type in ["Linear Regression", "Logistic Regression"]:
        df_plot = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Residual': y_true - y_pred,
            'Index': range(len(y_true))
        })
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add scatter plot for predictions
        scatter = go.Scatter(
            x=df_plot['Index'],
            y=df_plot['Actual'],
            mode='markers',
            name='Actual',
            marker=dict(
                size=8,
                color='#00ff00',
                opacity=0.6
            ),
            hovertemplate="<br>".join([
                "Index: %{x}",
                "Actual: %{y:.2f}",
                "<extra></extra>"
            ])
        )
        fig.add_trace(scatter)
        
        # Add predicted values
        pred_scatter = go.Scatter(
            x=df_plot['Index'],
            y=df_plot['Predicted'],
            mode='lines+markers',
            name='Predicted',
            marker=dict(
                size=8,
                color='#ff9900',
                opacity=0.6
            ),
            line=dict(
                color='#ff9900',
                width=2,
                dash='dot'
            ),
            hovertemplate="<br>".join([
                "Index: %{x}",
                "Predicted: %{y:.2f}",
                "<extra></extra>"
            ])
        )
        fig.add_trace(pred_scatter)
        
        # Add residuals on secondary y-axis
        residual_scatter = go.Scatter(
            x=df_plot['Index'],
            y=df_plot['Residual'],
            mode='markers',
            name='Residual',
            marker=dict(
                size=6,
                color='#ff0000',
                opacity=0.4
            ),
            hovertemplate="<br>".join([
                "Index: %{x}",
                "Residual: %{y:.2f}",
                "<extra></extra>"
            ]),
            yaxis="y2"
        )
        fig.add_trace(residual_scatter, secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title="Actual vs Predicted Values",
            template="plotly_dark",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Sample Index")
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text="Residual", secondary_y=True)
        
        return fig

def create_cluster_plot(X, labels, centers):
    """Create an interactive scatter plot for K-Means clustering."""
    if X.shape[1] < 2:
        st.error("Need at least 2 features for cluster visualization")
        return None
        
    # Select first two features for visualization
    feature_1, feature_2 = X.columns[:2]
    
    df_plot = pd.DataFrame({
        feature_1: X[feature_1],
        feature_2: X[feature_2],
        'Cluster': labels
    })
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add data points
    for cluster in range(len(centers)):
        cluster_data = df_plot[df_plot['Cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data[feature_1],
            y=cluster_data[feature_2],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=8,
                opacity=0.6
            ),
            hovertemplate="<br>".join([
                f"{feature_1}: %{x:.2f}",
                f"{feature_2}: %{y:.2f}",
                "Cluster: %{text}",
                "<extra></extra>"
            ]),
            text=[f'Cluster {cluster}'] * len(cluster_data)
        ))
    
    # Add cluster centers
    fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(
            color='#ffffff',
            size=12,
            symbol='x',
            line=dict(
                color='#000000',
                width=2
            )
        ),
        hovertemplate="<br>".join([
            "Centroid",
            f"{feature_1}: %{x:.2f}",
            f"{feature_2}: %{y:.2f}",
            "<extra></extra>"
        ])
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Cluster Distribution ({feature_1} vs {feature_2})",
        template="plotly_dark",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text=feature_1)
    fig.update_yaxes(title_text=feature_2)
    
    return fig

def create_feature_importance_plot(model, feature_names, X, y):
    """Create feature importance visualization."""
    if isinstance(model, (LinearRegression, LogisticRegression)):
        # For linear models, use coefficients
        importance = np.abs(model.coef_)
        if importance.ndim > 1:  # For multi-class logistic regression
            importance = importance.mean(axis=0)
            
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='#00ff00'
        ))
        
        # Calculate permutation importance for comparison
        perm_importance = permutation_importance(
            model, X, y, n_repeats=5, random_state=42
        )
        
        # Add permutation importance
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=perm_importance.importances_mean,
            orientation='h',
            marker_color='#ff9900',
            opacity=0.7,
            name='Permutation Importance'
        ))
        
        fig.update_layout(
            title="Feature Importance Analysis",
            template="plotly_dark",
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            barmode='group'
        )
        
        fig.update_xaxes(title_text="Importance Score")
        fig.update_yaxes(title_text="Features")
        
        return fig
    return None

def get_table_download_link(df, filename="predictions.csv"):
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    # Add version info and deployment details
    st.sidebar.markdown("---")
    st.sidebar.markdown("### App Info")
    st.sidebar.text("Version: 1.0.0")
    st.sidebar.text("Last Updated: 2024-02")
    
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.text(f"Python: {sys.version.split()[0]}")
        st.sidebar.text(f"Streamlit: {st.__version__}")
        st.sidebar.text(f"Pandas: {pd.__version__}")
        st.sidebar.text(f"Scikit-learn: {sklearn.__version__}")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Step:",
        ["Data Input", "Data Preprocessing", "Model Training", "Evaluation"]
    )

    # Data Input Section
    if page == "Data Input":
        st.title("Data Input")
        
        input_method = st.sidebar.selectbox(
            "Choose Input Method",
            ["Upload CSV", "Yahoo Finance Data"]
        )

        if input_method == "Upload CSV":
            st.write("### Upload CSV Data")
            uploaded_file = st.file_uploader(
                "Upload your CSV file", 
                type=['csv'],
                help="Upload a CSV file containing your dataset"
            )
            
            if uploaded_file is not None:
                df = load_csv_data(uploaded_file)
                if df is not None:
                    st.write("### Data Preview")
                    st.dataframe(
                        df.head(),
                        use_container_width=True,
                        column_config={col: st.column_config.Column(
                            help=f"Type: {df[col].dtype}"
                        ) for col in df.columns}
                    )
                    st.write(f"Total rows: {len(df)}")
                    st.write(f"Total columns: {len(df.columns)}")

        else:  # Yahoo Finance Data
            st.write("### Yahoo Finance Data")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                ticker = st.text_input(
                    "Enter Stock Ticker Symbol",
                    placeholder="e.g., AAPL",
                    help="Enter a valid stock ticker symbol"
                )
            
            with col2:
                # Date range selection
                today = datetime.today()
                default_start = today - timedelta(days=365)  # 1 year ago
                
                dates = st.date_input(
                    "Select Date Range",
                    value=(default_start, today),
                    max_value=today,
                    help="Select the date range for stock data"
                )

            if ticker and len(dates) == 2:
                start_date, end_date = dates
                df = load_stock_data(ticker.upper(), start_date, end_date)
                
                if df is not None:
                    st.write("### Data Preview")
                    st.dataframe(
                        df.head(),
                        use_container_width=True,
                        column_config={col: st.column_config.Column(
                            help=f"Type: {df[col].dtype}"
                        ) for col in df.columns}
                    )
                    st.write(f"Total trading days: {len(df)}")
                    
                    # Display basic statistics
                    st.write("### Basic Statistics")
                    st.dataframe(
                        df.describe(),
                        use_container_width=True
                    )

    # Data Preprocessing Section
    elif page == "Data Preprocessing":
        st.title("Data Preprocessing")
        
        if st.session_state.data is None:
            st.warning("Please load data in the Data Input section first.")
            return
            
        df = st.session_state.data
        
        st.write("### Missing Values Analysis")
        
        # Display missing values summary
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing == 0:
            st.success("No missing values found in the dataset!")
        else:
            st.write(f"Total missing values: {total_missing}")
            
            # Plot missing values
            fig = plot_missing_values(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Missing values handling
            st.write("### Handle Missing Values")
            
            # Get columns with missing values
            columns_with_missing = missing_values[missing_values > 0].index.tolist()
            
            if columns_with_missing:
                st.write("Select imputation strategy for each column:")
                
                # Create a dictionary to store strategies for each column
                strategies = {}
                
                for col in columns_with_missing:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**{col}**")
                        st.write(f"Missing: {missing_values[col]} ({(missing_values[col]/len(df)*100):.1f}%)")
                    
                    with col2:
                        if df[col].dtype in [np.number, float, int]:
                            strategy = st.selectbox(
                                f"Strategy for {col}",
                                ['mean', 'median', 'zero', 'drop'],
                                key=f"strategy_{col}"
                            )
                        else:
                            strategy = st.selectbox(
                                f"Strategy for {col}",
                                ['mode', 'empty', 'drop'],
                                key=f"strategy_{col}"
                            )
                        strategies[col] = strategy
                
                if st.button("Apply Preprocessing"):
                    processed_df = handle_missing_values(df, strategies)
                    st.session_state.processed_data = processed_df
                    
                    st.success("Preprocessing completed successfully!")
                    
                    st.write("### Processed Data Preview")
                    st.dataframe(
                        processed_df.head(),
                        use_container_width=True
                    )
                    
                    # Show shape comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Shape:", df.shape)
                    with col2:
                        st.write("Processed Shape:", processed_df.shape)
                    
                    # Verify no missing values remain
                    remaining_missing = processed_df.isnull().sum().sum()
                    if remaining_missing == 0:
                        st.success("All missing values have been handled!")
                    else:
                        st.warning(f"There are still {remaining_missing} missing values in the dataset.")

    # Model Training Section
    elif page == "Model Training":
        st.title("Model Training")
        
        if st.session_state.processed_data is None:
            if st.session_state.data is None:
                st.warning("Please load data in the Data Input section first.")
                return
            df = st.session_state.data
            st.info("Using original data as no preprocessing has been applied.")
        else:
            df = st.session_state.processed_data
            st.success("Using preprocessed data for model training.")
        
        # Add auto-refresh information for Yahoo Finance data
        if 'last_refresh' in df.attrs:
            st.info(f"Data last refreshed: {df.attrs['last_refresh']}")
            
            # Calculate time until next refresh
            last_refresh = datetime.strptime(df.attrs['last_refresh'], '%Y-%m-%d %H:%M:%S')
            next_refresh = last_refresh + timedelta(hours=1)
            time_to_refresh = next_refresh - datetime.now()
            
            if time_to_refresh.total_seconds() > 0:
                st.write(f"Next auto-refresh in: {int(time_to_refresh.total_seconds() / 60)} minutes")
        
        st.write("### Feature Selection")
        
        # Get numeric columns for feature selection
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("The dataset needs at least 2 numeric columns for modeling.")
            return
        
        # Feature selection
        selected_features = st.multiselect(
            "Select features for training:",
            numeric_columns,
            default=numeric_columns[:-1],
            help="Select the columns to use as features for training"
        )
        
        if not selected_features:
            st.warning("Please select at least one feature.")
            return
        
        # Target variable selection
        remaining_columns = [col for col in numeric_columns if col not in selected_features]
        if not remaining_columns:
            st.error("No numeric columns left for target variable selection.")
            return
            
        target_variable = st.selectbox(
            "Select target variable:",
            remaining_columns,
            help="Select the column to predict"
        )
        
        # Model selection
        st.write("### Model Configuration")
        
        model_type = st.selectbox(
            "Select model type:",
            ["Linear Regression", "Logistic Regression", "K-Means"],
            help="Choose the type of model to train"
        )
        
        # Model parameters
        test_size = st.slider(
            "Test set size:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.1,
            help="Proportion of the dataset to include in the test split"
        )
        
        # Training button
        if st.button("Train Model"):
            with st.spinner("Training model... Please wait."):
                try:
                    # Prepare features and target
                    X = df[selected_features]
                    y = df[target_variable] if model_type != "K-Means" else None
                    
                    # Train model
                    model, metrics, scaler = train_model(
                        X, y, 
                        model_type=model_type,
                        test_size=test_size
                    )
                    
                    # Store in session state
                    st.session_state.trained_model = model
                    st.session_state.model_metrics = metrics
                    st.session_state.feature_scaler = scaler
                    
                    # Display results
                    st.success("Model training completed!")
                    
                    # Display metrics based on model type
                    st.write("### Training Results")
                    if model_type == "Linear Regression":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                        with col2:
                            st.metric("MSE", f"{metrics['mse']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    
                    elif model_type == "Logistic Regression":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        with col2:
                            st.metric("Correct Predictions", 
                                    f"{metrics['correct_predictions']}/{metrics['test_size']}")
                    
                    else:  # K-Means
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Inertia", f"{metrics['inertia']:.4f}")
                        with col2:
                            st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
                    
                    # Save model info
                    st.write("### Model Information")
                    st.json({
                        'model_type': model_type,
                        'features': selected_features,
                        'target': target_variable if model_type != "K-Means" else "N/A",
                        'test_size': test_size
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")

    # Evaluation Section
    elif page == "Evaluation":
        st.title("Model Evaluation")
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model first in the Model Training section.")
            return
            
        st.write("### Model Visualization")
        
        # Get the model and data
        model = st.session_state.trained_model
        model_type = type(model).__name__
        
        if isinstance(model, (LinearRegression, LogisticRegression)):
            # Get original data
            if st.session_state.processed_data is not None:
                df = st.session_state.processed_data
            else:
                df = st.session_state.data
                
            # Get features and target
            X = df[st.session_state.model_metrics['features']]
            y = df[st.session_state.model_metrics['target']]
            
            # Scale features
            X_scaled = st.session_state.feature_scaler.transform(X)
            
            # Get predictions
            y_pred = model.predict(X_scaled)
            st.session_state.predictions = y_pred  # Store predictions
            
            # Create prediction plot
            fig = create_prediction_plot(y, y_pred, model_type)
            
            # Create feature importance plot
            importance_fig = create_feature_importance_plot(
                model, 
                st.session_state.model_metrics['features'],
                X_scaled,
                y
            )
            
            if importance_fig:
                st.write("### Feature Importance")
                st.plotly_chart(importance_fig, use_container_width=True)
            
        elif isinstance(model, KMeans):
            # Get scaled features
            X = df[st.session_state.model_metrics['features']]
            X_scaled = st.session_state.feature_scaler.transform(X)
            
            # Create cluster plot
            fig = create_cluster_plot(
                pd.DataFrame(X_scaled, columns=X.columns),
                model.labels_,
                model.cluster_centers_
            )
        
        if fig is not None:
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Plot as PNG",
                    data=fig.to_image(format="png", engine="kaleido"),
                    file_name="model_visualization.png",
                    mime="image/png"
                )
            
            with col2:
                if st.session_state.predictions is not None:
                    # Create predictions DataFrame
                    pred_df = pd.DataFrame({
                        'Actual': y,
                        'Predicted': st.session_state.predictions,
                        'Difference': y - st.session_state.predictions
                    })
                    
                    # Add download button for predictions
                    csv = pred_df.to_csv(index=True)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            
            # Display interactive features info
            st.info("""
            💡 **Interactive Features:**
            - Zoom: Click and drag to zoom, double-click to reset
            - Pan: Click and drag while zoomed
            - Hover: Mouse over points for details
            - Legend: Click to toggle traces
            """)
        
        # Display model metrics with dynamic selection
        st.write("### Model Metrics")
        if isinstance(model, LinearRegression):
            metric_options = ['R² Score', 'MSE', 'RMSE']
            selected_metric = st.selectbox("Select metric to display:", metric_options)
            
            if selected_metric == 'R² Score':
                st.metric("R² Score", f"{st.session_state.model_metrics['r2_score']:.4f}")
            elif selected_metric == 'MSE':
                st.metric("MSE", f"{st.session_state.model_metrics['mse']:.4f}")
            else:
                st.metric("RMSE", f"{st.session_state.model_metrics['rmse']:.4f}")
                
        elif isinstance(model, LogisticRegression):
            st.metric("Accuracy", f"{st.session_state.model_metrics['accuracy']:.4f}")
            st.metric("Correct Predictions", 
                     f"{st.session_state.model_metrics['correct_predictions']}/{st.session_state.model_metrics['test_size']}")
        
        else:  # KMeans
            st.metric("Inertia", f"{st.session_state.model_metrics['inertia']:.4f}")
            st.metric("Silhouette Score", f"{st.session_state.model_metrics['silhouette']:.4f}")

if __name__ == "__main__":
    main() 