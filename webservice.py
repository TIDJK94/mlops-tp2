from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn
import os

app = FastAPI()

# Global model variable
current_model = None
current_model_name = None

@app.on_event("startup")
async def startup_event():
    """Load default model on startup"""
    global current_model, current_model_name
    
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Try to load default model
    try:
        current_model_name = "digimon_hp_predictor_random_forest"
        model_uri = f"models:/{current_model_name}/3"  # Use latest version
        current_model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model: {current_model_name} version 3")
    except Exception as e:
        print(f"Could not load default model: {e}")
        # Try version 1 as fallback
        try:
            model_uri = f"models:/{current_model_name}/1"
            current_model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded fallback model: {current_model_name} version 1")
        except Exception as e2:
            print(f"Could not load fallback model: {e2}")

@app.get("/")
async def root():
    return {"message": "MLFlow Model Service", "model_loaded": current_model is not None}

@app.post("/predict")
async def predict(data: dict):
    """Make predictions"""
    if not current_model:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    try:
        # Convert to DataFrame and predict
        input_df = pd.DataFrame([data])
        prediction = current_model.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update-model")
async def update_model(request: dict):
    """Update the loaded model"""
    global current_model, current_model_name
    
    model_name = request.get("model_name")
    version = request.get("version", "1")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required")
    
    try:
        model_uri = f"models:/{model_name}/{version}"
        current_model = mlflow.sklearn.load_model(model_uri)
        current_model_name = model_name
        return {"message": f"Model updated to {model_name} version {version}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)