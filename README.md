# Financial ML Pipeline

A Streamlit web application for financial data analysis and machine learning pipeline.

## Features

- Real-time stock data fetching using Yahoo Finance API
- Data preprocessing and feature engineering
- Multiple ML models support:
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
- Interactive visualizations using Plotly
- Downloadable results and predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-ml-pipeline.git
cd financial-ml-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your own dataset or fetch real-time stock data
2. Choose a machine learning model
3. Follow the step-by-step pipeline:
   - Data preprocessing
   - Feature engineering
   - Model training
   - Evaluation and visualization

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Project Structure

- `app.py`: Main application file
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore file
- `README.md`: Project documentation

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- matplotlib
- yfinance
- scikit-learn
- pillow
- base64

## 🎨 Design Features

- **Color Scheme:** Emerald (#009B77) & Gold (#FFD700)
- **Font:** Playfair Display (serif)
- **UI Elements:** 
  - Vintage stock certificate style borders
  - Animated coin-stack progress indicators
  - Themed GIFs at key pages

## 🏆 Objective

This application allows students to build a complete Machine Learning pipeline using financial data. They can upload Kragle datasets, fetch real-time stock prices via Yahoo Finance API, choose/train/evaluate ML models, and visualize the results at each stage.

## 📚 Learning Outcomes

- **LO3:** Develop financial decision-making models and algorithms
- **LO5:** Visualize & interpret financial data effectively
- **LO8:** Demonstrate self-learning to enhance finance programming skills

## 📋 Features

- **Data Sources:**
  - Upload CSV/XLSX from "Kragle" datasets
  - Fetch real-time stock data via `yfinance`
  
- **Models:**
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
  
- **Pipeline Steps:**
  - Load Data
  - Preprocessing
  - Feature Engineering
  - Train/Test Split
  - Model Training
  - Evaluation
  - Results Visualization
  
- **Extras:**
  - Interactive Plotly charts
  - Download functionality for data and results
  - Error handling for missing data

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install via pip)

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Generate sample data (optional):
   ```
   python financial_ml_notebook.py
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
5. Open the application in your browser at http://localhost:8501

## 📊 Sample Datasets

- `financial_ml_notebook.py` includes code to generate:
  - Real stock data from Yahoo Finance
  - Synthetic "Kragle" financial dataset

## 📱 App Structure

1. **Welcome Interface**
   - Finance-themed GIF + welcome banner
   - Sidebar with data upload/fetch options
   
2. **ML Pipeline Steps**
   - Each step has its own button to execute
   - Success animations after each step
   - Visualizations of results
   
3. **Results & Downloads**
   - Interactive charts and visualizations
   - Download options for processed data and results

## 📚 Technologies Used

- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Plotly
- yfinance 