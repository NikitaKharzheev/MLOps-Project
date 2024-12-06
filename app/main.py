from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ml_service import ModelManager
from s3_service import (
    upload_to_s3,
    download_from_s3,
    get_list_from_bucket,
    create_bucket,
)
import psutil
import json
import os
import logging
import tempfile

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    filename="fastapi_log.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
model_manager = ModelManager()
try:
    create_bucket()
except Exception as e:
    logger.error(f"Failed to create MinIO bucket: {e}")


class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: Optional[Dict] = {}
    target_variable: str


class UpdateRequest(BaseModel):
    model_type: str
    hyperparameters: Optional[Dict] = {}
    target_variable: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    available_memory: str
    used_memory: str


@app.post("/upload-data")
async def upload_data(data: UploadFile = File(...)):
    """
    Uploads a JSON file to MinIO.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(await data.read())
            tmp_file_path = tmp_file.name

        upload_to_s3(tmp_file_path, f"data/{data.filename}")

    finally:
        if tmp_file_path:
            os.remove(tmp_file_path)

    return {"message": f"File {data.filename} uploaded successfully"}


@app.post("/train")
async def train_model(train_request: TrainRequest):
    """
    Trains a model using data stored in MinIO.
    """
    logger.info("Received training request")
    data_key = "data/iris.json"
    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file_path = tmp_file.name

        download_from_s3(data_key, tmp_file_path)

        with open(tmp_file_path, "r") as f:
            data_content = json.load(f)

        model_id = model_manager.train_model(
            model_type=train_request.model_type,
            hyperparameters=train_request.hyperparameters,
            data_content=data_content,
            target_variable=train_request.target_variable,
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".joblib"
        ) as model_tmp_file:
            model_tmp_path = model_tmp_file.name
            model_manager.save_model(model_id, model_tmp_path)

        upload_to_s3(model_tmp_path, f"models/{model_id}.joblib")
        os.remove(model_tmp_path)

    finally:
        if tmp_file_path:
            os.remove(tmp_file_path)

    logger.info(f"Model {model_id} trained and saved to MinIO")
    return {"message": "Training completed", "model_id": model_id}


@app.get("/models", response_model=List[str])
async def list_available_models():
    """
    Retrieves a list of available machine learning models.

    This endpoint provides a list of model types that can be trained and used
    for predictions. The response is a list of strings, where each string
    represents a model type.

    :return: A JSON response containing a list of available model types.
    """
    logger.info("Listing available models")
    return model_manager.get_available_models()


@app.post("/predict/{model_id}")
async def predict(model_id: str, data: UploadFile = File(...)):
    """
    Makes predictions using a specified model.
    """
    logger.info(f"Received prediction request for model {model_id}")
    model_key = f"models/{model_id}.joblib"
    tmp_model_path = None
    tmp_data_path = None

    try:
        # Создаём временный файл для модели
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".joblib"
        ) as model_tmp_file:
            tmp_model_path = model_tmp_file.name

        # Скачиваем модель из MinIO
        download_from_s3(model_key, tmp_model_path)

        # Создаём временный файл для данных
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as data_tmp_file:
            tmp_data_path = data_tmp_file.name
            data_tmp_file.write(await data.read())

        # Читаем данные из временного файла
        with open(tmp_data_path, "r") as f:
            data_content = json.load(f)

        # Делаем предсказания
        prediction = model_manager.predict(model_id, data_content, tmp_model_path)

    finally:
        # Удаляем временные файлы
        if tmp_model_path:
            os.remove(tmp_model_path)
        if tmp_data_path:
            os.remove(tmp_data_path)

    return {"prediction": prediction}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Deletes a trained model from MinIO.
    """
    logger.info(f"Received request to delete model {model_id}")
    model_key = f"models/{model_id}.joblib"
    try:
        from s3_service import s3_client, S3_BUCKET_NAME

        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=model_key)
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=404, detail="Model not found")
    logger.info(f"Model {model_id} deleted from MinIO")
    return {"message": f"Model {model_id} deleted successfully"}


@app.get("/status", response_model=StatusResponse)
async def status():
    """
    Returns service status and system memory usage.
    """
    memory_info = psutil.virtual_memory()
    available_memory = f"{memory_info.available / (1024 ** 2):.2f} MB"
    used_memory = f"{memory_info.used / (1024 ** 2):.2f} MB"
    return StatusResponse(
        status="Service is running",
        available_memory=available_memory,
        used_memory=used_memory,
    )


@app.get("/trained-models")
async def list_trained_models():
    logger.info("Listing trained models")
    try:
        trained_models = get_list_from_bucket("models/")
    except Exception as e:
        raise HTTPException(status_code=404, detail="No trained models found")
    return {"trained_models": trained_models}


@app.get("/prediction/{model_id}")
async def get_predictions(model_id: str):
    """
    Retrieves the saved predictions for a given model ID.

    Args:
        model_id: The unique identifier of the model.

    Returns:
        A JSON response containing the saved predictions.

    Raises:
        HTTPException: If the model ID is not found in MinIO or
            no predictions are available.
    """
    logger.info(f"Fetching saved predictions for model {model_id}")
    model_key = f"predictions/{model_id}"
    local_predictions_path = f"/tmp/{model_id}"

    # Download predictions from MinIO
    try:
        download_from_s3(model_key, local_predictions_path)
    except Exception as e:
        logger.error(f"Prediction for model {model_id} not found in MinIO: {e}")
        raise HTTPException(status_code=404, detail="Prediction not found")

    with open(local_predictions_path, "r") as f:
        saved_predictions = json.load(f)
    os.remove(local_predictions_path)

    if saved_predictions is None:
        logger.error(f"Predictions not found for model {model_id}")
        raise HTTPException(
            status_code=404, detail="Model not found or no predictions available"
        )

    return saved_predictions


@app.put("/update-model/{model_id}")
async def update_model(model_id: str, update_request: UpdateRequest):
    logger.info(f"Received update request for model {model_id}")
    data_key = "data/iris.json"  # Change to dynamic if needed
    local_data_path = f"/tmp/{os.path.basename(data_key)}"

    try:
        download_from_s3(data_key, local_data_path)
    except Exception as e:
        logger.error(f"Failed to download dataset from MinIO: {e}")
        raise HTTPException(status_code=400, detail="Dataset not found in MinIO")

    # Read data from the downloaded file
    with open(local_data_path, "r") as f:
        data_content = json.load(f)
    os.remove(local_data_path)

    # Train the model
    model_id, trained_model = model_manager.update_model(
        model_id=model_id,
        model_type=update_request.model_type,
        hyperparameters=update_request.hyperparameters,
        data_content=data_content,
        target_variable=update_request.target_variable,
    )

    # Save model to MinIO
    model_path = f"/tmp/{model_id}.joblib"
    model_manager.save_model(trained_model, model_path)
    upload_to_s3(model_path, f"models/{model_id}.joblib")
    os.remove(model_path)
    logger.info(f"Model {model_id} updated and saved to MinIO")
    return {"message": "Updating completed", "model_id": model_id}
