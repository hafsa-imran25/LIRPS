import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_loan_dataset(n_samples=1000):
    """Generate synthetic loan dataset for interest rate prediction"""
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    # Define possible values
    loan_purposes = ['home', 'auto', 'personal', 'business', 'education', 'medical']
    employment_types = ['full_time', 'part_time', 'self_employed', 'unemployed', 'retired']
    education_levels = ['high_school', 'bachelor', 'master', 'phd', 'associates']
    
    for i in range(n_samples):
        # Basic demographics
        age = np.random.normal(40, 12)
        age = max(18, min(80, age))  # Clamp between 18 and 80
        
        # Income based on age and education
        education = random.choice(education_levels)
        education_multiplier = {
            'high_school': 0.8,
            'associates': 0.9,
            'bachelor': 1.0,
            'master': 1.3,
            'phd': 1.5
        }[education]
        
        base_income = 30000 + (age - 18) * 1000 * education_multiplier
        income = max(15000, np.random.normal(base_income, 15000))
        
        # Employment
        employment = random.choice(employment_types)
        if employment == 'unemployed':
            income *= 0.1  # Unemployment benefits
        elif employment == 'part_time':
            income *= 0.6
        elif employment == 'retired':
            income *= 0.7
        
        # Credit score (300-850)
        credit_base = 650 + (income - 40000) / 1000
        credit_score = max(300, min(850, np.random.normal(credit_base, 80)))
        
        # Loan amount
        loan_purpose = random.choice(loan_purposes)
        if loan_purpose == 'home':
            loan_amount = np.random.normal(200000, 100000)
        elif loan_purpose == 'auto':
            loan_amount = np.random.normal(25000, 10000)
        elif loan_purpose == 'business':
            loan_amount = np.random.normal(50000, 30000)
        else:
            loan_amount = np.random.normal(15000, 8000)
        
        loan_amount = max(1000, loan_amount)
        
        # Loan term (months)
        if loan_purpose == 'home':
            loan_term = random.choice([180, 240, 300, 360])  # 15-30 years
        elif loan_purpose == 'auto':
            loan_term = random.choice([36, 48, 60, 72])  # 3-6 years
        else:
            loan_term = random.choice([12, 24, 36, 48, 60])  # 1-5 years
        
        # Debt to income ratio
        existing_debt = np.random.normal(income * 0.3, income * 0.15)
        existing_debt = max(0, existing_debt)
        debt_to_income = (existing_debt + (loan_amount / loan_term * 12)) / income
        debt_to_income = min(1.0, debt_to_income)
        
        # Years of credit history
        credit_history_years = max(0, min(age - 18, np.random.normal(age - 25, 5)))
        
        # Number of previous loans
        previous_loans = max(0, int(np.random.poisson(credit_history_years / 5)))
        
        # Calculate interest rate based on multiple factors
        base_rate = 5.0  # Base interest rate
        
        # Credit score impact (most important)
        if credit_score >= 750:
            credit_adjustment = -1.5
        elif credit_score >= 700:
            credit_adjustment = -0.5
        elif credit_score >= 650:
            credit_adjustment = 0
        elif credit_score >= 600:
            credit_adjustment = 1.0
        else:
            credit_adjustment = 3.0
        
        # Debt-to-income impact
        if debt_to_income <= 0.28:
            dti_adjustment = -0.5
        elif debt_to_income <= 0.36:
            dti_adjustment = 0
        elif debt_to_income <= 0.45:
            dti_adjustment = 1.0
        else:
            dti_adjustment = 2.5
        
        # Loan purpose impact
        purpose_adjustment = {
            'home': -0.5,
            'auto': 0,
            'education': -0.3,
            'personal': 1.5,
            'business': 1.0,
            'medical': 0.8
        }[loan_purpose]
        
        # Employment stability
        employment_adjustment = {
            'full_time': 0,
            'part_time': 0.5,
            'self_employed': 1.0,
            'unemployed': 5.0,
            'retired': 0.2
        }[employment]
        
        # Loan amount impact (larger loans get better rates)
        if loan_amount >= 100000:
            amount_adjustment = -0.3
        elif loan_amount >= 50000:
            amount_adjustment = 0
        else:
            amount_adjustment = 0.5
        
        # Credit history impact
        if credit_history_years >= 10:
            history_adjustment = -0.3
        elif credit_history_years >= 5:
            history_adjustment = 0
        else:
            history_adjustment = 0.5
        
        # Calculate final interest rate
        interest_rate = (base_rate + credit_adjustment + dti_adjustment + 
                        purpose_adjustment + employment_adjustment + 
                        amount_adjustment + history_adjustment)
        
        # Add some random noise
        interest_rate += np.random.normal(0, 0.5)
        
        # Ensure reasonable bounds
        interest_rate = max(2.0, min(25.0, interest_rate))
        
        # Create record
        record = {
            'age': round(age, 1),
            'income': round(income, 2),
            'credit_score': round(credit_score, 0),
            'loan_amount': round(loan_amount, 2),
            'loan_term_months': loan_term,
            'loan_purpose': loan_purpose,
            'employment_type': employment,
            'education_level': education,
            'debt_to_income_ratio': round(debt_to_income, 3),
            'credit_history_years': round(credit_history_years, 1),
            'previous_loans': previous_loans,
            'interest_rate': round(interest_rate, 2)
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('loan_dataset.csv', index=False)
    
    print(f"Generated dataset with {n_samples} samples")
    print(f"Interest rate range: {df['interest_rate'].min():.2f}% - {df['interest_rate'].max():.2f}%")
    print(f"Dataset saved as 'loan_dataset.csv'")
    print("\nDataset preview:")
    print(df.head())
    print("\nDataset info:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_loan_dataset(2000)