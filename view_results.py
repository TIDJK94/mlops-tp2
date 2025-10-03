#!/usr/bin/env python3
"""
MLflow Results Viewer
====================

Simple script to view and compare MLflow experiment results.
"""

import mlflow
import pandas as pd

def view_experiment_results():
    """View results from all experiments"""
    print("🔍 MLflow Experiment Results")
    print("=" * 50)
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        if exp.name == "Default":
            continue
            
        print(f"\n📊 Experiment: {exp.name}")
        print(f"   ID: {exp.experiment_id}")
        
        # Get runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs.empty:
            print("   No runs found.")
            continue
            
        print(f"   Total runs: {len(runs)}")
        
        # Display key metrics and parameters
        for idx, run in runs.iterrows():
            print(f"\\n   🏃 Run {idx + 1}:")
            print(f"      Run ID: {run['run_id']}")
            print(f"      Status: {run['status']}")
            
            # Parameters
            print(f"      📋 Parameters:")
            param_cols = [col for col in run.index if col.startswith('params.')]
            for param_col in param_cols:
                param_name = param_col.replace('params.', '')
                if pd.notna(run[param_col]):
                    print(f"         {param_name}: {run[param_col]}")
            
            # Metrics
            print(f"      📈 Metrics:")
            metric_cols = [col for col in run.index if col.startswith('metrics.')]
            for metric_col in metric_cols:
                metric_name = metric_col.replace('metrics.', '')
                if pd.notna(run[metric_col]):
                    if 'r2' in metric_name.lower():
                        print(f"         {metric_name}: {run[metric_col]:.3f}")
                    else:
                        print(f"         {metric_name}: {run[metric_col]:.2f}")
    
    print("\\n" + "=" * 50)
    print("💡 To start MLflow UI, run: mlflow ui")
    print("   Then open http://localhost:5000 in your browser")

def compare_models():
    """Compare models across experiments"""
    print("\\n🔬 Model Comparison")
    print("=" * 30)
    
    all_runs = []
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        if exp.name == "Default":
            continue
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for idx, run in runs.iterrows():
            run_info = {
                'experiment': exp.name,
                'model_type': run.get('params.model_type', 'unknown'),
                'test_rmse': run.get('metrics.test_rmse', None),
                'test_r2': run.get('metrics.test_r2', None),
                'test_mae': run.get('metrics.test_mae', None)
            }
            all_runs.append(run_info)
    
    if not all_runs:
        print("No runs found for comparison.")
        return
    
    # Sort by test RMSE (lower is better)
    all_runs.sort(key=lambda x: x['test_rmse'] if x['test_rmse'] is not None else float('inf'))
    
    print("📊 Models ranked by Test RMSE (lower is better):")
    for i, run in enumerate(all_runs, 1):
        print(f"\\n{i}. {run['model_type']} ({run['experiment']})")
        if run['test_rmse'] is not None:
            print(f"   Test RMSE: {run['test_rmse']:.2f}")
        if run['test_r2'] is not None:
            print(f"   Test R²:   {run['test_r2']:.3f}")
        if run['test_mae'] is not None:
            print(f"   Test MAE:  {run['test_mae']:.2f}")

if __name__ == "__main__":
    view_experiment_results()
    compare_models()