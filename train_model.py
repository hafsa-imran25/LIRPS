import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_interest_rate_model():
    """Train machine learning model for interest rate prediction"""
    
    # Load the dataset
    try:
        df = pd.read_csv('loan_dataset.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset not found! Please run generate_loan_dataset.py first.")
        return
    
    # Data preprocessing
    print("\n=== Data Preprocessing ===")
    
    # Handle categorical variables
    categorical_columns = ['loan_purpose', 'employment_type', 'education_level']
    
    # Create label encoders
    label_encoders = {}
    df_processed = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Select features for modeling
    feature_columns = [
        'age', 'income', 'credit_score', 'loan_amount', 'loan_term_months',
        'debt_to_income_ratio', 'credit_history_years', 'previous_loans',
        'loan_purpose_encoded', 'employment_type_encoded', 'education_level_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['interest_rate']
    
    print(f"\nFeatures: {feature_columns}")
    print(f"Target: interest_rate")
    print(f"Feature matrix shape: {X.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'income', 'credit_score', 'loan_amount', 
                         'debt_to_income_ratio', 'credit_history_years']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Train multiple models
    print("\n=== Model Training ===")
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = float('inf')
    best_model_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_model_name = name
    
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Best RMSE: {best_score:.4f}")
    
    # Feature importance (for Random Forest)
    if best_model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
    
    # Save the best model and preprocessing objects
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'numerical_features': numerical_features,
        'best_model_name': best_model_name,
        'model_metrics': {
            'rmse': best_score,
            'mae': results[best_model_name]['mae'],
            'r2': results[best_model_name]['r2']
        }
    }
    
    joblib.dump(model_artifacts, 'loan_interest_model.pkl')
    print(f"\nModel saved as 'loan_interest_model.pkl'")
    
    # Create prediction function for testing
    def predict_interest_rate(age, income, credit_score, loan_amount, loan_term_months,
                            loan_purpose, employment_type, education_level,
                            debt_to_income_ratio, credit_history_years, previous_loans):
        """Function to predict interest rate for new data"""
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'credit_score': [credit_score],
            'loan_amount': [loan_amount],
            'loan_term_months': [loan_term_months],
            'debt_to_income_ratio': [debt_to_income_ratio],
            'credit_history_years': [credit_history_years],
            'previous_loans': [previous_loans],
            'loan_purpose_encoded': [label_encoders['loan_purpose'].transform([loan_purpose])[0]],
            'employment_type_encoded': [label_encoders['employment_type'].transform([employment_type])[0]],
            'education_level_encoded': [label_encoders['education_level'].transform([education_level])[0]]
        })
        
        # Scale numerical features if needed
        if best_model_name != 'Random Forest':
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        
        # Make prediction
        prediction = best_model.predict(input_data[feature_columns])[0]
        return round(prediction, 2)
    
    # Test the prediction function
    print("\n=== Testing Prediction Function ===")
    test_prediction = predict_interest_rate(
        age=35, income=75000, credit_score=720, loan_amount=25000,
        loan_term_months=60, loan_purpose='auto', employment_type='full_time',
        education_level='bachelor', debt_to_income_ratio=0.25,
        credit_history_years=10, previous_loans=2
    )
    print(f"Sample prediction: {test_prediction}%")
    
    return model_artifacts

if __name__ == "__main__":
    train_interest_rate_model()