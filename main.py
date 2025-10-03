import kagglehub
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os

def train_and_register_model():
    """Train a model and register it in MLFlow"""
    
    # Set MLFlow tracking URI to local directory
    tracking_uri = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    
    # Set environment variables to avoid artifact proxy issues
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"file://{tracking_uri}"
    
    print("Loading Digimon dataset from local file...")
    # Use local dataset file
    dataset_path = "/home/tidjk/scia/mlops/model-versioning/DigiDB_digimonlist.csv"
    
    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Prepare features for HP prediction using actual column names
    feature_columns = ['Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']
    target_column = 'Lv 50 HP'
    
    # Check if all required columns exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features or target_column not in df.columns:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Clean data
    df_clean = df[available_features + [target_column]].dropna()
    
    X = df_clean[available_features]
    y = df_clean[target_column]
    
    print(f"Features used: {available_features}")
    print(f"Final dataset shape: {df_clean.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLFlow run
    with mlflow.start_run(run_name="digimon_hp_predictor") as run:
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log model (without registered name for now to avoid artifact issues)
        model_name = "digimon_hp_predictor_random_forest"
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained and registered as: {model_name}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"MLFlow Run ID: {run.info.run_id}")
        
        return model_name, run.info.run_id

if __name__ == "__main__":
    print("Starting MLFlow model training...")
    model_name, run_id = train_and_register_model()
    print(f"Training completed! Model '{model_name}' is ready to be served.")
    print("Start the web service with: python webservice.py")
