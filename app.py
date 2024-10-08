import pickle
from functools import wraps

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import joinedload, sessionmaker

app = Flask(__name__)

from models import *

# Database setup
DATABASE_URL = 'sqlite:///database/db.db'
engine = create_engine(DATABASE_URL)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)

app.secret_key = 'Secret_Key'

# Registration and Login routes for users
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        db_session = DBSession()
        existing_user = db_session.query(user).filter_by(username=username).first()
        if existing_user:
            db_session.close()
            return "Username already exists. Please choose a different one."

        # Create a new user
        new_user = user(username=username, password=password)
        db_session.add(new_user)
        db_session.commit()
        db_session.close()
        return redirect('/login')

    return render_template('user_register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db_session = DBSession()
        user_entry = db_session.query(user).filter_by(username=username, password=password).first()
        db_session.close()

        if user_entry:
            # Store user info in the session
            session['user_id'] = user_entry.userid
            return redirect('/')
        else:
            return "Login failed. Please check your username and password."

    return render_template('user_login.html')


# Route for logging out
@app.route('/logout')
def logout():
    # Clear the user session
    session.clear()
    # Redirect to the index page or any other desired page after logout
    return redirect('/')

# Load the fitted scaler and ensemble model for Depression
with open("standard_scalerdep.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("depressionensemble_model.pkl", "rb") as ensemble_model_file:
    ensemble_model = pickle.load(ensemble_model_file)


# Load the fitted scaler and ensemble model for Cardiovascular
with open("cardioscaler.pkl", "rb") as scaler_file:
    cardioscaler = pickle.load(scaler_file)

with open("cardioensemble_model.pkl", "rb") as ensemble_model_file:
    cardioensemble_model = pickle.load(ensemble_model_file)
    

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Render the Depression Page
@app.route('/depression')
@login_required
def depression():
    return render_template('depression.html')

# Render the Depression Page
@app.route('/cardiovascular')
@login_required
def cardiovascular():
    return render_template('cardiovascular.html')

# Handle the cardiovascular prediction
@app.route('/predict-cardiovascular', methods=['POST'])
def predict_cardiovascular():
    if request.method == 'POST':
        # Get the input features from the form
        features = [
            int(request.form['age']),
            int(request.form['gender']),
            int(request.form['height']),
            float(request.form['weight']),
            int(request.form['ap_hi']),
            int(request.form['ap_lo']),
            int(request.form['cholesterol']),
            int(request.form['gluc']),
            int(request.form['smoke']),
            int(request.form['alco']),
            int(request.form['active'])
        ]

        # Scale the input features
        cardioscaled_features = cardioscaler.transform([features])

        # Make a prediction using the cardiovascular model
        prediction = cardioensemble_model.predict(cardioscaled_features)

        # Display the prediction result
        result = "Positive" if prediction[0] == 1 else "Negative"

        return render_template('cardiovascular.html', result=result)

# Handle the prediction
@app.route('/predictdepression', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input features from the form
        features = [
            int(request.form['sex']),
            int(request.form['Age']),
            int(request.form['Married']),
            int(request.form['Number_children']),
            int(request.form['education_level']),
            int(request.form['total_members']),
            int(request.form['gained_asset']),
            int(request.form['durable_asset']),
            int(request.form['save_asset']),
            int(request.form['living_expenses']),
            int(request.form['other_expenses']),
            int(request.form['incoming_salary']),
            int(request.form['incoming_own_farm']),
            int(request.form['incoming_business']),
            int(request.form['incoming_no_business']),
            int(request.form['incoming_agricultural']),
            int(request.form['farm_expenses']),
            int(request.form['labor_primary']),
            int(request.form['lasting_investment']),
            float(request.form['no_lasting_investmen'])
        ]

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Make a prediction using the ensemble model
        prediction = ensemble_model.predict(scaled_features)

        # Display the prediction result
        result = "Depressed" if prediction[0] == 1 else "Not Depressed"

        return render_template('depression.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
