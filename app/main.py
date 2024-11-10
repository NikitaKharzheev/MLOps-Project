from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ml_service import ModelManager
import json

app = FastAPI()
model_manager = ModelManager()

class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: Optional[Dict] = {}
    target_variable: str

class PredictRequest(BaseModel):
    model_id: str

class UpdateRequest(BaseModel):
    hyperparameters: Optional[Dict] = {}
    target_variable: Optional[str] = None


class StatusResponse(BaseModel):
    status: str

uploaded_data_storage = {}

@app.post("/upload-data")
async def upload_data(data: UploadFile = File(...)):
    # if data.content_type != "application/json":
    #     raise HTTPException(status_code=400, detail="File must be a JSON")
    
    contents = await data.read()
    try:
        data_content = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    
   
    uploaded_data_storage["last_uploaded_data"] = data_content
    return {"message": "Data uploaded successfully", "data_preview": data_content[:5]}

@app.post("/train")
async def train_model(train_request: TrainRequest):
    data_content = uploaded_data_storage.get("last_uploaded_data")
    if not data_content:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first using /upload-data")
    
    if train_request.model_type not in model_manager.get_available_models():
        raise HTTPException(status_code=400, detail="Model type not supported")

    model_id = model_manager.train_model(
        model_type=train_request.model_type,
        hyperparameters=train_request.hyperparameters,
        data_content=data_content,
        target_variable=train_request.target_variable
    )
    return {"message": "Training started", "model_id": model_id}

@app.get("/models", response_model=List[str])
async def list_available_models():
    return model_manager.get_available_models()

@app.post("/predict/{model_id}")
async def predict(model_id: str, data: UploadFile = File(...)):
    data_json = await data.read()
    data_content = json.loads(data_json)
    
    prediction = model_manager.predict(model_id, data_content)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Model not found or not trained")
    
    return {"prediction": prediction}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    result = model_manager.delete_model(model_id)
    if result:
        return {"message": "Model deleted"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(status="Service is running")


@app.get("/trained-models")
async def list_trained_models():
    trained_models = model_manager.get_trained_models()
    return {"trained_models": trained_models}


@app.get("/prediction/{model_id}")
async def get_predictions(model_id: str):
    
    saved_predictions = model_manager.get_predictions(model_id)
    if saved_predictions is None:
        raise HTTPException(status_code=404, detail="Model not found or no predictions available")

    return {"predictions": saved_predictions}


@app.put("/update-model/{model_id}")
async def update_model(model_id: str, update_request: UpdateRequest):
   
    data_content = uploaded_data_storage.get("last_uploaded_data")
    result = model_manager.update_model(
        model_id=model_id,
        data_content=data_content,
        hyperparameters=update_request.hyperparameters,
        target_variable=update_request.target_variable
    )
    if result:
        return {"message": "Model updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")