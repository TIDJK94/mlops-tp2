# MLflow Model Deployment

MLflow model training, web service deployment, and canary deployment for Digimon HP prediction.

## Quick Start

### Train Model
```bash
python mlflow_training.py
```

### Run Web Service
```bash
python webservice.py
```

### Docker Deployment
```bash
docker-compose up --build
```

## Web Service Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Predict
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Memory": 6,
    "Equip Slots": 2,
    "Lv50 SP": 78,
    "Lv50 Atk": 131,
    "Lv50 Def": 99,
    "Lv50 Int": 95,
    "Lv50 Spd": 112,
    "Stage_encoded": 5,
    "Type_encoded": 2,
    "Attribute_encoded": 1
  }'
```

### Update Model (Canary)
```bash
curl -X POST "http://localhost:8000/update-model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "digimon_hp_predictor_random_forest", "version": "3"}'
```

### Accept Next Model
```bash
curl -X POST "http://localhost:8000/accept-next-model"
```

## Canary Deployment

The service supports canary deployment with:
- Current model (90% traffic)
- Next model (10% traffic) 
- Ability to update next model independently
- Promotion of next model to current

## Files

- `mlflow_training.py` - Train and register models
- `webservice.py` - FastAPI web service with canary deployment
- `DigiDB_digimonlist.csv` - Dataset
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Docker deployment