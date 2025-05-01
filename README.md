# ML Pipeline Streamlit App 🚀

An interactive machine learning pipeline application built with Streamlit. This app allows users to load data, preprocess it, train various models, and visualize results with an intuitive interface.

## 🌟 Features

- **Data Loading**
  - CSV file upload support
  - Yahoo Finance integration with auto-refresh
  - Real-time data validation

- **Data Preprocessing**
  - Missing value detection and visualization
  - Multiple imputation strategies
  - Interactive data preview

- **Model Training**
  - Support for multiple algorithms:
    - Linear Regression
    - Logistic Regression
    - K-Means Clustering
  - Feature selection
  - Automatic train-test split

- **Visualization & Evaluation**
  - Interactive Plotly charts
  - Feature importance analysis
  - Model performance metrics
  - Downloadable predictions
  - Dark theme UI

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd streamlit_ml_app
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## 📊 Data Format

### CSV Files
- Must contain numeric columns for modeling
- Should not have missing column names
- Recommended to have clean column names (no spaces/special characters)

### Yahoo Finance
- Enter valid ticker symbols (e.g., AAPL, GOOGL)
- Data automatically refreshes every 60 minutes
- Historical data available from 1970 onwards

## 🔧 Model Configuration

### Linear Regression
- Best for continuous target variables
- Provides R², MSE, and RMSE metrics
- Features importance visualization

### Logistic Regression
- Suitable for binary classification
- Shows accuracy and prediction counts
- Supports feature importance analysis

### K-Means Clustering
- Automatic cluster number detection
- Silhouette score evaluation
- Interactive cluster visualization

## ☁️ Deployment

### Streamlit Cloud Deployment
1. Create a Streamlit account at https://streamlit.io
2. Connect your GitHub repository
3. Deploy with these settings:
   - Main file path: `app.py`
   - Python version: 3.8+
   - Requirements: `requirements.txt`

### Local Deployment
For local deployment, ensure:
1. All dependencies are installed
2. Sufficient RAM (recommended: 4GB+)
3. Internet connection for Yahoo Finance data

## 🔒 Security Notes

- No API keys required
- Data is processed locally
- No data persistence between sessions
- User data is not stored

## 🐛 Troubleshooting

Common issues and solutions:

1. **Yahoo Finance Connection Error**
   - Check internet connection
   - Verify ticker symbol
   - Wait a few minutes and retry

2. **Model Training Errors**
   - Ensure numeric data only
   - Check for missing values
   - Verify feature/target selection

3. **Visualization Issues**
   - Clear browser cache
   - Update plotly package
   - Check data size

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📧 Support

For support, please open an issue in the GitHub repository. 