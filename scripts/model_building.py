# scripts/model_building.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

def load_data(fraud_data_path, creditcard_data_path):
    # Load the datasets
    fraud_data = pd.read_csv(fraud_data_path)
    creditcard_data = pd.read_csv(creditcard_data_path)
    return fraud_data, creditcard_data

def preprocess_data(fraud_data, creditcard_data):
    # Separate features and target
    X_fraud = fraud_data.drop(columns=['class'])
    y_fraud = fraud_data['class']
    
    X_creditcard = creditcard_data.drop(columns=['Class'])
    y_creditcard = creditcard_data['Class']
    
    return X_fraud, y_fraud, X_creditcard, y_creditcard

def train_models(X_train, y_train):
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'MLP Classifier': MLPClassifier(max_iter=500)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    evaluation_results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        evaluation_results[name] = {
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return evaluation_results

def main(fraud_data_path, creditcard_data_path):
    # Load data
    fraud_data, creditcard_data = load_data(fraud_data_path, creditcard_data_path)
    
    # Preprocess data
    X_fraud, y_fraud, X_creditcard, y_creditcard = preprocess_data(fraud_data, creditcard_data)
    
    # Train-test split for both datasets
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(X_creditcard, y_creditcard, test_size=0.2, random_state=42)
    
    # Train models
    trained_models_fraud = train_models(X_train_fraud, y_train_fraud)
    trained_models_creditcard = train_models(X_train_creditcard, y_train_creditcard)
    
    # Evaluate models
    evaluation_results_fraud = evaluate_models(trained_models_fraud, X_test_fraud, y_test_fraud)
    evaluation_results_creditcard = evaluate_models(trained_models_creditcard, X_test_creditcard, y_test_creditcard)
    
    return evaluation_results_fraud, evaluation_results_creditcard

if __name__ == '__main__':
    fraud_data_path = 'path/to/Fraud_Data.csv'  # Update with your path
    creditcard_data_path = 'path/to/creditcard.csv'  # Update with your path
    results_fraud, results_creditcard = main(fraud_data_path, creditcard_data_path)
    print("Fraud Data Evaluation Results:")
    for model, result in results_fraud.items():
        print(f"{model}:\n{result['report']}\nConfusion Matrix:\n{result['confusion_matrix']}\n")
    
    print("Credit Card Data Evaluation Results:")
    for model, result in results_creditcard.items():
        print(f"{model}:\n{result['report']}\nConfusion Matrix:\n{result['confusion_matrix']}\n")
