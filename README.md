# Health-Predictor-A-Flask-Web-Application-for-Depression-and-Cardiovascular-Disease-Prediction

## Project Overview

**Health Predictor** is a Flask-based web application that integrates machine learning models to predict the likelihood of two major health conditions: **depression** and **cardiovascular diseases**. The goal is to provide users with early health insights based on their inputs, helping them monitor risks and seek timely interventions.

## Project Details

This project addresses global health concerns regarding cardiovascular diseases and mental health issues like depression. Cardiovascular diseases are one of the leading causes of death worldwide, while depression affects millions globally. By utilizing machine learning in a simple web interface, the Health Predictor application offers accessible tools to assess and predict these conditions.

The main objectives of the project are:
1. **Cardiovascular Disease Prediction**: Using machine learning models such as logistic regression, Random Forest, and Gradient Boosting. The ensemble approach (Voting Classifier) achieved an accuracy of **73%**.
2. **Depression Prediction**: Random Forest and other algorithms were used, with Random Forest achieving an accuracy of **83.22%**.
3. **Flask Web Application**: Integration of the models in a user-friendly web interface for easy health assessments.

The prediction models are trained using datasets from **Kaggle**:
- [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- [Depression Dataset](https://www.kaggle.com/datasets/diegobabativa/depression)

## Features

- **Depression Prediction**: Based on user-provided socio-economic and health-related data, the model assesses mental health risks.
- **Cardiovascular Disease Prediction**: Predicts cardiovascular risks using health metrics like blood pressure, cholesterol levels, and lifestyle factors.
- **User Authentication**: The app includes secure login and registration, ensuring personalized predictions.
- **Results Visualization**: Clearly presents prediction results with supporting metrics like accuracy, precision, and recall.

## Installation

### Prerequisites

- **Python 3.x**
- **Pip** (Python package installer)
- **Flask** for the web application
- **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, **Seaborn**, and **Pickle** for machine learning and data handling.

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/Vamshigouni/health-predictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Health-Predictor-A-Flask-Web-Application-for-Depression-and-Cardiovascular-Disease-Prediction
    ```
3. Install the required dependencies:
    - Ensure that the following packages are installed in your environment:
      - Flask
      - pandas
      - numpy
      - scikit-learn
      - matplotlib
      - seaborn
      - pickle

    - You can install these packages individually using:
    ```bash
    pip install Flask pandas numpy scikit-learn matplotlib seaborn pickle
    ```
4. Download the necessary datasets from Kaggle and update the dataset paths in `depression.py` and `cardio.py` files:
    - [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
    - [Depression Dataset](https://www.kaggle.com/datasets/diegobabativa/depression)

### Running the Application

1. Run the **Depression Code.py**:
    ```bash
    python depression.py
    ```
    This will generate accuracy, precision, recall, and F1 score metrics.

2. Run the **Cardio-Vascular Code.py**:
    ```bash
    python cardio.py
    ```
    The output will include similar metrics and save the model using the Pickle library.

3. To start the web application:
    ```bash
    python app.py
    ```
    Open the generated URL in a web browser to access the application.

### Application Workflow

1. **Home Page**: Users can choose between **Depression** or **Cardiovascular Disease** prediction.
2. **Login/Registration**: Users need to register or log in to access the health prediction tools.
3. **Prediction Inputs**: Users enter health metrics (e.g., age, blood pressure) for cardiovascular risk, or socio-economic details for depression prediction.
4. **Results**: The application provides detailed predictions based on user inputs, including accuracy, recall, and precision metrics.

## Project Architecture

The backend of the application is built using Flask and SQLAlchemy, providing a database for user authentication. The machine learning models are integrated using the **Pickle** library to load pre-trained models for fast predictions.

## Model Performance

- **Cardiovascular Disease Prediction**:
    - Logistic Regression: 72% accuracy
    - Voting Classifier (Random Forest + Gradient Boosting): 73% accuracy
- **Depression Prediction**:
    - Random Forest: 83.22% accuracy

## Future Work

- Improve model accuracy through hyperparameter tuning and the inclusion of additional features.
- Extend the application to predict other common health issues.
- Explore commercialization opportunities for the app in collaboration with healthcare providers.
