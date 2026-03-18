# Australian Rainfall Prediction Classifier

## Project Overview

This project develops a binary classification model to predict whether it will rain tomorrow in Australia based on historical weather data. The model utilizes machine learning techniques to analyze various meteorological features and provide accurate rainfall predictions.

The project is implemented in a Jupyter Notebook and includes data preprocessing, feature engineering, model training with hyperparameter tuning, evaluation, and a simple web interface for deployment using Streamlit.

## Dataset Description

The dataset used is the "Weather Dataset" from Kaggle, containing daily weather observations from multiple locations across Australia. It includes 23 features such as temperature, humidity, wind speed, pressure, and rainfall measurements collected over several years.

**Key Features:**
- Date and location information
- Temperature metrics (min, max, 9am, 3pm)
- Humidity and pressure readings
- Wind speed and direction
- Rainfall amounts
- Cloud cover and sunshine hours

**Target Variable:**
- RainTomorrow: Binary classification (Yes/No) indicating whether it rained the next day

**Dataset Source:** https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv

## Methodology

### 1. Data Cleaning
- Handled missing values using `dropna()` to remove rows with null values
- Removed duplicate entries to ensure data quality
- Performed initial exploratory data analysis to understand data distribution

### 2. Feature Engineering
- Created a 'Season' feature derived from date information to capture seasonal patterns
- Renamed the target column from 'RainTomorrow' to 'RainToday' to prevent data leakage during training
- Identified and separated numerical and categorical features for appropriate preprocessing

### 3. Preprocessing
- Utilized `ColumnTransformer` and `Pipeline` for streamlined preprocessing
- Applied `StandardScaler` for numerical feature normalization
- Implemented `OneHotEncoder` for categorical variable encoding
- Ensured consistent preprocessing across training and testing datasets

### 4. Modeling
- Trained two classification models:
  - **RandomForestClassifier**: Ensemble method for robust predictions
  - **LogisticRegression**: Linear model for interpretability
- Employed `GridSearchCV` with `StratifiedKFold` for hyperparameter tuning
- Optimized model parameters to maximize performance metrics

### 5. Evaluation
- Analyzed model performance using:
  - **Confusion Matrix**: Visual representation of prediction accuracy
  - **Classification Report**: Detailed metrics including precision, recall, and F1-score
  - **Feature Importance Plots**: Identified key predictors for rainfall

### 6. Deployment
- Built a simple web interface using Streamlit
- Created an interactive application for real-time predictions
- Enabled user-friendly input of weather parameters for rainfall forecasting

## Results Interpretation

### Model Performance
The models achieved the following key metrics on the test dataset:

- **Accuracy**: [Insert specific accuracy values from your notebook]
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the ability to identify all positive cases
- **F1-Score**: Harmonic mean of precision and recall

### Key Findings
- **Top Predictive Features**: Humidity, pressure, and temperature variables showed the highest importance
- **Model Comparison**: RandomForest generally outperformed LogisticRegression in handling complex patterns
- **Seasonal Patterns**: Certain seasons showed higher prediction accuracy due to distinct weather patterns

### Confusion Matrix Analysis
- True Positives: Correctly predicted rainfall days
- True Negatives: Correctly predicted non-rainfall days
- False Positives: Incorrectly predicted rainfall (Type I error)
- False Negatives: Missed rainfall predictions (Type II error)

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Step-by-Step Setup Guide

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd aus-rainfall-prediction
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv rainfall_env
   source rainfall_env/bin/activate  # On Windows: rainfall_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open `AUS_Weather_Predict_Project.ipynb` and run all cells to train the models.

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   Access the web interface at `http://localhost:8501`

## Usage

### Training the Model
1. Open the Jupyter Notebook
2. Execute cells sequentially to:
   - Load and preprocess data
   - Train RandomForest and LogisticRegression models
   - Evaluate performance with visualizations

### Making Predictions
1. Launch the Streamlit application
2. Input current weather parameters
3. Receive rainfall prediction with confidence score

## Project Structure
```
aus-rainfall-prediction/
│
├── AUS_Weather_Predict_Project.ipynb    # Main training notebook
├── app.py                                # Streamlit web application
├── requirements.txt                      # Python dependencies
├── README.md                            # Project documentation
└── data/                                # Dataset storage (if local)
```

## Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Web application framework
- **Jupyter Notebook**: Interactive development environment

## Future Improvements
- Implement additional machine learning models (XGBoost, Neural Networks)
- Add more sophisticated feature engineering techniques
- Deploy the model to cloud platforms (AWS, GCP)
- Create API endpoints for integration with other applications
- Implement time series forecasting for multi-day predictions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by Kaggle
- Inspired by various machine learning tutorials and courses
- Thanks to the open-source community for excellent libraries