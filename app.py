from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variable to store model artifacts
model_artifacts = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model_artifacts
    try:
        model_artifacts = joblib.load('loan_interest_model.pkl')
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model file not found! Please train the model first.")
        return False

def predict_interest_rate(input_data):
    """Predict interest rate for given input data"""
    global model_artifacts
    
    if model_artifacts is None:
        return None, "Model not loaded"
    
    try:
        # Extract model components
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        label_encoders = model_artifacts['label_encoders']
        feature_columns = model_artifacts['feature_columns']
        numerical_features = model_artifacts['numerical_features']
        best_model_name = model_artifacts['best_model_name']
        
        # Create input dataframe
        df_input = pd.DataFrame({
            'age': [float(input_data['age'])],
            'income': [float(input_data['income'])],
            'credit_score': [float(input_data['credit_score'])],
            'loan_amount': [float(input_data['loan_amount'])],
            'loan_term_months': [int(input_data['loan_term_months'])],
            'debt_to_income_ratio': [float(input_data['debt_to_income_ratio'])],
            'credit_history_years': [float(input_data['credit_history_years'])],
            'previous_loans': [int(input_data['previous_loans'])],
            'loan_purpose_encoded': [label_encoders['loan_purpose'].transform([input_data['loan_purpose']])[0]],
            'employment_type_encoded': [label_encoders['employment_type'].transform([input_data['employment_type']])[0]],
            'education_level_encoded': [label_encoders['education_level'].transform([input_data['education_level']])[0]]
        })
        
        # Scale numerical features if needed
        if best_model_name != 'Random Forest':
            df_input[numerical_features] = scaler.transform(df_input[numerical_features])
        
        # Make prediction
        prediction = model.predict(df_input[feature_columns])[0]
        
        # Calculate additional metrics
        monthly_payment = calculate_monthly_payment(
            float(input_data['loan_amount']), 
            prediction, 
            int(input_data['loan_term_months'])
        )
        
        total_interest = (monthly_payment * int(input_data['loan_term_months'])) - float(input_data['loan_amount'])
        
        return {
            'interest_rate': round(prediction, 2),
            'monthly_payment': round(monthly_payment, 2),
            'total_interest': round(total_interest, 2),
            'total_payment': round(monthly_payment * int(input_data['loan_term_months']), 2)
        }, None
        
    except Exception as e:
        return None, str(e)

def calculate_monthly_payment(principal, annual_rate, months):
    """Calculate monthly payment using loan formula"""
    if annual_rate == 0:
        return principal / months
    
    monthly_rate = annual_rate / 100 / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return payment

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        input_data = {
            'age': request.form['age'],
            'income': request.form['income'],
            'credit_score': request.form['credit_score'],
            'loan_amount': request.form['loan_amount'],
            'loan_term_months': request.form['loan_term_months'],
            'loan_purpose': request.form['loan_purpose'],
            'employment_type': request.form['employment_type'],
            'education_level': request.form['education_level'],
            'debt_to_income_ratio': request.form['debt_to_income_ratio'],
            'credit_history_years': request.form['credit_history_years'],
            'previous_loans': request.form['previous_loans']
        }
        
        # Make prediction
        result, error = predict_interest_rate(input_data)
        
        if error:
            return jsonify({'success': False, 'error': error})
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    """Get model information"""
    global model_artifacts
    
    if model_artifacts is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    metrics = model_artifacts.get('model_metrics', {})
    return jsonify({
        'success': True,
        'model_name': model_artifacts.get('best_model_name', 'Unknown'),
        'metrics': metrics
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Cannot start application without model. Please run:")
        print("1. python generate_loan_dataset.py")
        print("2. python train_model.py")
        print("3. python app.py")