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

        model_id, trained_model = model_manager.train_model(
            model_type=train_request.model_type,
            hyperparameters=train_request.hyperparameters,
            data_content=data_content,
            target_variable=train_request.target_variable,
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".joblib"
        ) as model_tmp_file:
            model_tmp_path = model_tmp_file.name
            model_manager.save_model(trained_model, model_tmp_path)

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


import tempfile


@app.post("/predict/{model_id}")
async def predict(model_id: str, data: UploadFile = File(...)):
    logger.info(f"Received prediction request for model {model_id}")
    model_key = f"models/{model_id}.joblib"

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.joblib")
        data_path = os.path.join(temp_dir, "data.json")
        prediction_path = os.path.join(temp_dir, "prediction.json")

        try:
            # Скачиваем модель из MinIO
            download_from_s3(model_key, model_path)

            # Сохраняем входные данные
            with open(data_path, "wb") as data_file:
                data_file.write(await data.read())

            # Читаем данные для предсказания
            with open(data_path, "r") as f:
                data_content = json.load(f)

            # Делаем предсказания
            prediction = model_manager.predict(model_id, data_content, model_path)

            # Сохраняем предсказания
            with open(prediction_path, "w") as pred_file:
                json.dump({f"{model_id}": prediction}, pred_file)

            # Загружаем предсказания в MinIO
            upload_to_s3(prediction_path, f"predictions/{model_id}")

        except Exception as e:
            logger.error(f"Failed to process prediction: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(f"Prediction saved to MinIO for model {model_id}")
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


import tempfile
import os
from fastapi import HTTPException


@app.get("/prediction/{model_id}")
async def get_predictions(model_id: str):
    """
    Retrieves the saved predictions for a given model ID.
    """
    logger.info(f"Fetching saved predictions for model {model_id}")
    model_key = f"predictions/{model_id}"

    # Создаём временную директорию для предсказаний
    with tempfile.TemporaryDirectory() as temp_dir:
        prediction_path = os.path.join(temp_dir, "prediction.json")

        try:
            # Скачиваем предсказания из MinIO
            download_from_s3(model_key, prediction_path)

            # Читаем предсказания
            with open(prediction_path, "r") as f:
                saved_predictions = json.load(f)

        except Exception as e:
            logger.error(f"Prediction for model {model_id} not found in MinIO: {e}")
            raise HTTPException(status_code=404, detail="Prediction not found")

        if not saved_predictions:
            logger.error(f"Predictions not found for model {model_id}")
            raise HTTPException(
                status_code=404, detail="Model not found or no predictions available"
            )

    return saved_predictions


@app.put("/update-model/{model_id}")
async def update_model(model_id: str, update_request: UpdateRequest):
    """
    Updates an existing model using new data and hyperparameters.
    """
    logger.info(f"Received update request for model {model_id}")
    data_key = "data/iris.json"  # Это можно сделать динамическим

    # Создаём временную директорию для данных и модели
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, "data.json")
        model_path = os.path.join(temp_dir, "model.joblib")

        try:
            # Скачиваем данные из MinIO
            download_from_s3(data_key, data_path)

            # Читаем данные
            with open(data_path, "r") as f:
                data_content = json.load(f)

            # Обновляем модель
            model_id, trained_model = model_manager.update_model(
                model_id=model_id,
                model_type=update_request.model_type,
                hyperparameters=update_request.hyperparameters,
                data_content=data_content,
                target_variable=update_request.target_variable,
            )

            # Сохраняем модель во временный файл
            model_manager.save_model(trained_model, model_path)

            # Загружаем обновлённую модель в MinIO
            upload_to_s3(model_path, f"models/{model_id}.joblib")

        except Exception as e:
            logger.error(f"Failed to update model {model_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update model")

    logger.info(f"Model {model_id} updated and saved to MinIO")
    return {"message": "Updating completed", "model_id": model_id}
