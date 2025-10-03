from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn
import os
import random

app = FastAPI()

# Global model variables for canary deployment
current_model = None
next_model = None
current_model_name = None
next_model_name = None
canary_probability = 0.1  # 10% traffic to next model by default

@app.on_event("startup")
async def startup_event():
    """Load default models for canary deployment on startup"""
    global current_model, next_model, current_model_name, next_model_name
    
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Try to load default model for both current and next
    try:
        current_model_name = "digimon_hp_predictor_random_forest"
        next_model_name = current_model_name
        model_uri = f"models:/{current_model_name}/4"  # Use latest version
        current_model = mlflow.sklearn.load_model(model_uri)
        next_model = mlflow.sklearn.load_model(model_uri)  # Same model initially
        print(f"Loaded current model: {current_model_name} version 4")
        print(f"Loaded next model: {next_model_name} version 4 (same as current)")
    except Exception as e:
        print(f"Could not load default model: {e}")
        # Try version 3 as fallback
        try:
            model_uri = f"models:/{current_model_name}/3"
            current_model = mlflow.sklearn.load_model(model_uri)
            next_model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded fallback models: {current_model_name} version 3")
        except Exception as e2:
            print(f"Could not load fallback model: {e2}")

@app.get("/")
async def root():
    return {
        "message": "MLFlow Canary Deployment Service", 
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "canary_probability": canary_probability
    }

@app.post("/predict")
async def predict(data: dict):
    """Make predictions using canary deployment"""
    if not current_model or not next_model:
        raise HTTPException(status_code=404, detail="Models not loaded")
    
    try:
        # Decide which model to use based on canary probability
        use_next_model = random.random() < canary_probability
        selected_model = next_model if use_next_model else current_model
        model_type = "next" if use_next_model else "current"
        
        # Convert to DataFrame and predict
        input_df = pd.DataFrame([data])
        prediction = selected_model.predict(input_df)
        return {
            "prediction": float(prediction[0]),
            "model_used": model_type
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update-model")
async def update_model(request: dict):
    """Update the next model for canary deployment"""
    global next_model, next_model_name
    
    model_name = request.get("model_name")
    version = request.get("version", "1")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required")
    
    try:
        model_uri = f"models:/{model_name}/{version}"
        next_model = mlflow.sklearn.load_model(model_uri)
        next_model_name = model_name
        return {"message": f"Next model updated to {model_name} version {version}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/accept-next-model")
async def accept_next_model():
    """Accept the next model as current (promote canary model)"""
    global current_model, next_model, current_model_name, next_model_name
    
    if not next_model:
        raise HTTPException(status_code=404, detail="No next model loaded")
    
    try:
        # Promote next model to current
        current_model = next_model
        current_model_name = next_model_name
        return {"message": f"Next model promoted to current. Both models are now {current_model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)