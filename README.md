# Financial Machine Learning Dashboard Assignment

## Overview
Build an interactive Streamlit application that implements a machine learning pipeline for financial data analysis. The application should follow a corporate blue theme and provide a professional, dashboard-style interface for financial data analysis and visualization.

## Learning Outcomes
- **LO3:** Develop financial models and algorithms for decision-making
- **LO5:** Visualize and interpret financial data effectively using Python tools
- **LO8:** Demonstrate self-learning skills to enhance programming capabilities for finance

## Theme & Design Guidelines
- **Color Scheme:** Navy (#000080) + Silver (#C0C0C0)
- **Font:** Sans-serif (Roboto)
- **Visual Elements:**
  - Stock-ticker ribbon animation at the top
  - Office-building skyline divider between sections
  - Professional dashboard layout

## Requirements

### Data Sources
- Upload static datasets from Kragle
- Fetch real-time quotes with `yfinance`

### Machine Learning Models (Choose One)
- Linear Regression
- Logistic Regression
- K-Means Clustering

### Required Python Libraries
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- yfinance

## Task Description

### A. Welcome Interface
- Finance-themed welcome screen
- Custom background color and themed buttons
- Sidebar for data input:
  - Kragle dataset upload
  - Yahoo Finance data fetch

### B. Step-by-Step ML Pipeline
1. **Data Loading**
   - Upload or fetch data
   - Display data preview
   - Success notification

2. **Preprocessing**
   - Clean missing values
   - Handle outliers
   - Display preprocessing statistics

3. **Feature Engineering**
   - Select/transform features
   - Display feature importance

4. **Model Training**
   - Train/test split
   - Model fitting
   - Training progress indicators

5. **Results Visualization**
   - Model metrics
   - Interactive plots
   - Prediction visualizations

### C. Additional Features
- Interactive Plotly visualizations
- Themed GIFs and images
- Downloadable results
- Real-time data updates

## Expected Deliverables
1. Well-commented Jupyter Notebook (`.ipynb`)
2. Streamlit app runnable locally
3. *(Optional Bonus)* Public Streamlit Cloud deployment

## Bonus Points
- Dynamic model selection
- Feature importance visualization
- Real-time data auto-refresh
- Download button for results

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Grading Criteria
- Code quality and organization (30%)
- UI/UX implementation (20%)
- ML pipeline functionality (30%)
- Documentation and comments (20%)

## Submission Guidelines
1. Submit your Jupyter Notebook and Streamlit app code
2. Include a brief report explaining your implementation choices
3. *(Optional)* Include Streamlit Cloud deployment link

## Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) 