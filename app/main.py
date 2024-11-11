from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ml_service import ModelManager
import psutil 
import json
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
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
    available_memory: str
    used_memory: str

uploaded_data_storage = {}

@app.post("/upload-data")
async def upload_data(data: UploadFile = File(...)):
    logger.info("Received data upload request")
    # if data.content_type != "application/json":
    #     raise HTTPException(status_code=400, detail="File must be a JSON")
    
    contents = await data.read()
    try:
        data_content = json.loads(contents)
        logger.info("Data successfully parsed as JSON")
    except json.JSONDecodeError:
        logger.error("Failed to parse uploaded data as JSON")
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    uploaded_data_storage["last_uploaded_data"] = data_content
    logger.info("Data uploaded successfully and stored")
    return {"message": "Data uploaded successfully", "data_preview": data_content[:5]}

@app.post("/train")
async def train_model(train_request: TrainRequest):
    logger.info("Received model training request")
    data_content = uploaded_data_storage.get("last_uploaded_data")
    if not data_content:
        logger.error("No data uploaded before training request")
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first using /upload-data")
    
    if train_request.model_type not in model_manager.get_available_models():
        logger.error(f"Unsupported model type: {train_request.model_type}")
        raise HTTPException(status_code=400, detail="Model type not supported")

    model_id = model_manager.train_model(
        model_type=train_request.model_type,
        hyperparameters=train_request.hyperparameters,
        data_content=data_content,
        target_variable=train_request.target_variable
    )
    logger.info(f"Training started for model {model_id}")
    return {"message": "Training started", "model_id": model_id}

@app.get("/models", response_model=List[str])
async def list_available_models():
    logger.info("Listing available models")
    return model_manager.get_available_models()

@app.post("/predict/{model_id}")
async def predict(model_id: str, data: UploadFile = File(...)):
    logger.info(f"Received prediction request for model {model_id}")
    data_json = await data.read()
    data_content = json.loads(data_json)
    
    prediction = model_manager.predict(model_id, data_content)
    if prediction is None:
        logger.error(f"Model {model_id} not found or not trained")
        raise HTTPException(status_code=404, detail="Model not found or not trained")
    
    logger.info(f"Prediction generated for model {model_id}")
    return {"prediction": prediction}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    logger.info(f"Received request to delete model {model_id}")
    result = model_manager.delete_model(model_id)
    if result:
        logger.info(f"Model {model_id} deleted successfully")
        return {"message": "Model deleted"}
    else:
        logger.error(f"Model {model_id} not found")
        raise HTTPException(status_code=404, detail="Model not found")


@app.get("/status", response_model=StatusResponse)
async def status():
    memory_info = psutil.virtual_memory()
    available_memory = f"{memory_info.available / (1024 ** 2):.2f} MB"
    used_memory = f"{memory_info.used / (1024 ** 2):.2f} MB"
    
    return StatusResponse(
        status="Service is running",
        available_memory=available_memory,
        used_memory=used_memory
    )


@app.get("/trained-models")
async def list_trained_models():
    logger.info("Listing trained models")
    trained_models = model_manager.get_trained_models()
    return {"trained_models": trained_models}


@app.get("/prediction/{model_id}")
async def get_predictions(model_id: str):
    logger.info(f"Fetching saved predictions for model {model_id}")
    saved_predictions = model_manager.get_predictions(model_id)
    if saved_predictions is None:
        logger.error(f"Predictions not found for model {model_id}")
        raise HTTPException(status_code=404, detail="Model not found or no predictions available")

    return {"predictions": saved_predictions}


@app.put("/update-model/{model_id}")
async def update_model(model_id: str, update_request: UpdateRequest):
    logger.info(f"Received update request for model {model_id}")
    data_content = uploaded_data_storage.get("last_uploaded_data")
    result = model_manager.update_model(
        model_id=model_id,
        data_content=data_content,
        hyperparameters=update_request.hyperparameters,
        target_variable=update_request.target_variable
    )
    if result:
        logger.info(f"Model {model_id} updated successfully")
        return {"message": "Model updated successfully"}
    else:
        logger.error(f"Model {model_id} not found for update")
        raise HTTPException(status_code=404, detail="Model not found")