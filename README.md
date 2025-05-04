# Financial ML Pipeline

A Streamlit-based web application for financial data analysis and machine learning. This application provides a complete pipeline for financial data analysis, from data loading to model evaluation and visualization.

## Features

- Data loading from CSV/Excel files or Yahoo Finance
- Data preprocessing and cleaning
- Feature engineering
- Multiple ML models (Linear Regression, Logistic Regression, K-Means Clustering)
- Model evaluation and visualization
- Interactive plots and charts
- Downloadable results

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/financial-ml-pipeline.git
cd financial-ml-pipeline
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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