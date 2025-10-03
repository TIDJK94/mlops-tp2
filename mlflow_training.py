#!/usr/bin/env python3
"""
Simple MLflow Model Training Script for Digimon HP Prediction
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    """Train and log model with MLflow"""
    
    # Configure MLflow to use local file store without proxy
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("digimon_hp_prediction")
    
    print("Loading Digimon dataset...")
    df = pd.read_csv("DigiDB_digimonlist.csv")
    
    # Prepare features
    feature_columns = ['Memory', 'Equip Slots', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']
    categorical_columns = ['Stage', 'Type', 'Attribute']
    
    # Encode categorical variables
    df_encoded = df.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
        feature_columns.append(f'{col}_encoded')
    
    # Prepare X and y
    X = df_encoded[feature_columns]
    y = df_encoded['Lv 50 HP']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        print("Training RandomForest model...")
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        
        # Log metrics to MLflow
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        
        # Log parameters to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Log and register model
        try:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="digimon_hp_predictor_random_forest"
            )
            print("Model registered successfully!")
        except Exception as e:
            # Fallback: just log the model without registration
            print(f"Registration failed ({e}), logging model without registration")
            mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained successfully!")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model, test_rmse, test_r2

if __name__ == "__main__":
    train_model()