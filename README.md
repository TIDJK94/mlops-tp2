# MLflow Model Training Demo

This project demonstrates **model versioning and experiment tracking** using **MLflow** with a Digimon dataset. The scripts show how to track hyperparameters, metrics, and model artifacts effectively.

## 🚀 Quick Start

### 1. Install Dependencies

The required packages are already installed in the virtual environment:
- `mlflow` - Experiment tracking and model registry
- `kagglehub` - Dataset downloading
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `matplotlib`, `seaborn` - Visualization

### 2. Run Training Scripts

#### Simple Example (Recommended for beginners)
```bash
python simple_mlflow_example.py
```

#### Advanced Example with Custom Parameters
```bash
# Random Forest with custom hyperparameters
python mlflow_training.py --model_type random_forest --n_estimators 100 --max_depth 10

# Linear Regression
python mlflow_training.py --model_type linear_regression

# Custom test set size
python mlflow_training.py --test_size 0.3 --n_estimators 200
```

### 3. View Results
```bash
# Command-line results summary
python view_results.py

# MLflow Web UI (recommended)
mlflow ui
# Then open http://localhost:5000 in your browser
```

## 📊 What Gets Tracked

### Hyperparameters Tracked:
- **Model Type**: RandomForestRegressor or LinearRegression
- **Dataset Info**: Size, features, train/test split
- **Model Parameters**: n_estimators, max_depth, min_samples_split, etc.
- **Random State**: For reproducibility

### Metrics Tracked:
- **Mean Squared Error (MSE)**: Training and test
- **Root Mean Squared Error (RMSE)**: Training and test  
- **Mean Absolute Error (MAE)**: Training and test
- **R² Score**: Training and test (coefficient of determination)
- **Feature Importance**: Top 3 most important features

### Artifacts Logged:
- **Trained Model**: Serialized sklearn model
- **Visualizations**: Model evaluation plots
- **Model Metadata**: Feature names, encoders, model info

## 🎯 Problem Description

**Objective**: Predict Digimon HP (Hit Points) based on their characteristics

**Dataset**: Digimon Database from Kaggle (rtatman/digidb)
- **Target Variable**: `Lv 50 HP` (Hit Points at level 50)
- **Features Used**:
  - Numerical: Memory, Equip Slots, SP, Attack, Defense, Intelligence, Speed
  - Categorical: Stage, Type, Attribute (encoded)

**Model Types**: 
- Random Forest Regressor (ensemble method)
- Linear Regression (baseline)

## 📈 Results Summary

Based on our experiments:

| Model | Test RMSE | Test R² | Test MAE | Notes |
|-------|-----------|---------|----------|-------|
| Linear Regression | 139.06 | 0.796 | 111.82 | **Best performer** |
| Random Forest (depth=8) | 174.18 | 0.680 | 123.16 | Overfitting issues |
| Random Forest (depth=10) | 177.03 | 0.669 | 127.60 | More overfitting |

**Key Insights**:
- Linear Regression performed better than Random Forest on this dataset
- Random Forest showed signs of overfitting (high training R², lower test R²)
- Most important features: Memory, Defense, Attack

## 📁 File Structure

```
model-versioning/
├── main.py                    # Original dataset download script
├── simple_mlflow_example.py   # Basic MLflow tracking example
├── mlflow_training.py         # Advanced MLflow script with CLI args
├── view_results.py           # Results viewer and model comparison
├── mlruns/                   # MLflow tracking data
│   ├── experiments/
│   └── models/
└── README.md                 # This file
```

## 🔧 Advanced Usage

### Custom Hyperparameter Tuning
```bash
# Try different Random Forest configurations
python mlflow_training.py --n_estimators 200 --max_depth 5 --min_samples_split 10
python mlflow_training.py --n_estimators 50 --max_depth 15 --min_samples_split 2
```

### Multiple Experiments
```bash
# Create runs with different experiment names
python mlflow_training.py --experiment_name "hyperparameter_tuning" --n_estimators 150
```

## 📚 MLflow Features Demonstrated

1. **Experiment Organization**: Multiple experiments for different model types
2. **Parameter Tracking**: All hyperparameters automatically logged
3. **Metric Tracking**: Comprehensive evaluation metrics
4. **Model Registry**: Models saved and versioned automatically
5. **Artifact Storage**: Plots and metadata saved with each run
6. **Reproducibility**: Random seeds and environment info tracked

## 🌐 MLflow UI Navigation

After running `mlflow ui`, you can:

- **Compare Runs**: Side-by-side comparison of different models
- **View Metrics**: Interactive metric charts and trends  
- **Download Models**: Export trained models for deployment
- **View Artifacts**: Check visualizations and model files
- **Filter/Search**: Find specific runs by parameters or metrics

## 🎓 Learning Objectives Achieved

✅ **Hyperparameter Tracking**: All model parameters logged automatically  
✅ **Metric Tracking**: MSE, MAE, R² tracked for both training and test sets  
✅ **Model Versioning**: Different models registered with versions  
✅ **Experiment Organization**: Separate experiments for different approaches  
✅ **Reproducibility**: Random seeds and configurations tracked  
✅ **Visualization**: Model performance plots generated and saved  

This demo provides a solid foundation for MLflow-based experiment tracking in machine learning projects!