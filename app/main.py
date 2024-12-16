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
import subprocess
import shutil

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


def run_dvc_command(command: List[str]):
    """
    Utility function to run DVC commands.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"DVC Command succeeded: {' '.join(command)}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC Command failed: {' '.join(command)}\n{e.stderr}")
        raise HTTPException(status_code=500, detail=f"DVC command failed: {e.stderr}")


def clean_datasets_folder():
    """
    Удаляет все файлы и подпапки в папке datasets/.
    """
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
        os.makedirs(datasets_dir)
        logger.info("Cleaned up the datasets folder.")


@app.post("/upload-data")
async def upload_data(data: UploadFile = File(...)):
    """
    Uploads a dataset to DVC and S3, including its .dvc file.
    """
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        dataset_path = os.path.join(datasets_dir, data.filename)

        # Сохраняем файл локально
        with open(dataset_path, "wb") as f:
            f.write(await data.read())

        logger.info(f"Dataset saved locally at {dataset_path}")

        # Добавляем файл в DVC
        logger.info(f"Adding {dataset_path} to DVC")
        run_dvc_command(["dvc", "add", dataset_path])
        logger.info(f"Dataset {dataset_path} added to DVC")

        # Получаем путь к .dvc файлу
        dvc_file_path = f"{dataset_path}.dvc"

        # Push файла и его метаданных в удалённое хранилище
        run_dvc_command(["dvc", "push"])
        logger.info(f"Dataset {dataset_path} pushed to remote storage")

        # Загружаем .dvc файл в S3
        logger.info(f"Uploading {dvc_file_path} to S3")
        upload_to_s3(dvc_file_path, f"dvc/{os.path.basename(dvc_file_path)}")

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"DVC command failed: {e.stderr}")

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")

    finally:
        clean_datasets_folder()

    return {
        "message": "Dataset uploaded and versioned successfully",
        "dvc_file": f"dvc/{os.path.basename(dvc_file_path)}",
    }


@app.post("/train")
async def train_model(train_request: TrainRequest):
    """
    Trains a model using data stored in DVC.
    """
    logger.info("Received training request")

    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        dvc_file_path = os.path.join(datasets_dir, "iris.json.dvc")
        dataset_path = os.path.join(datasets_dir, "iris.json")
        model_path = os.path.join(datasets_dir, "model.joblib")
        dvc_file = "dvc/iris.json.dvc"

        # Скачиваем .dvc файл из S3
        logger.info(f"Downloading .dvc file {dvc_file} from S3")
        download_from_s3(dvc_file, dvc_file_path)
        logger.info(f".dvc file {dvc_file} downloaded successfully")

        # Используем DVC для загрузки датасета
        logger.info("Pulling dataset from remote storage using DVC")
        run_dvc_command(["dvc", "pull"])
        logger.info("Dataset pulled successfully from DVC")

        # Читаем данные
        with open(dataset_path, "r") as f:
            data_content = json.load(f)

        # Обучение модели
        model_id, trained_model = model_manager.train_model(
            model_type=train_request.model_type,
            hyperparameters=train_request.hyperparameters,
            data_content=data_content,
            target_variable=train_request.target_variable,
        )

        # Сохраняем обученную модель
        model_manager.save_model(trained_model, model_path)

        # Загружаем модель в S3
        logger.info(f"Uploading trained model {model_id} to S3")
        upload_to_s3(model_path, f"models/{model_id}.joblib")

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"DVC command failed: {e.stderr}")

    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train model")

    finally:
        clean_datasets_folder()

    logger.info(f"Model {model_id} trained and saved to S3")
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
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        model_path = os.path.join(datasets_dir, "model.joblib")
        data_path = os.path.join(datasets_dir, "data.json")
        prediction_path = os.path.join(datasets_dir, "prediction.json")

        # Скачиваем модель из MinIO
        download_from_s3(model_key, model_path)
        logger.info(f"Model {model_id} downloaded successfully")

        # Сохраняем входные данные во временный файл
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

    finally:
        clean_datasets_folder()

    logger.info(f"Prediction for model {model_id} saved to MinIO")
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
    except Exception:
        raise HTTPException(status_code=404, detail="No trained models found")
    return {"trained_models": trained_models}


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

    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        dvc_file_path = os.path.join(datasets_dir, "iris.json.dvc")
        dataset_path = os.path.join(datasets_dir, "iris.json")
        model_path = os.path.join(datasets_dir, "updated_model.joblib")
        dvc_file = "dvc/iris.json.dvc"

        # Скачиваем .dvc файл из S3
        logger.info(f"Downloading .dvc file {dvc_file} from S3")
        download_from_s3(dvc_file, dvc_file_path)
        logger.info(f".dvc file {dvc_file} downloaded successfully")

        # Используем DVC для загрузки датасета
        logger.info("Pulling dataset from remote storage using DVC")
        run_dvc_command(["dvc", "pull"])
        logger.info("Dataset pulled successfully from DVC")

        # Читаем данные
        with open(dataset_path, "r") as f:
            data_content = json.load(f)

        # Обновляем модель
        updated_model_id, updated_model = model_manager.update_model(
            model_id=model_id,
            model_type=update_request.model_type,
            hyperparameters=update_request.hyperparameters,
            data_content=data_content,
            target_variable=update_request.target_variable,
        )

        # Сохраняем обновлённую модель
        model_manager.save_model(updated_model, model_path)

        # Загружаем обновлённую модель в MinIO
        upload_to_s3(model_path, f"models/{updated_model_id}.joblib")

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"DVC command failed: {e.stderr}")

    except Exception as e:
        logger.error(f"Failed to update model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")

    finally:
        clean_datasets_folder()

    logger.info(f"Model {updated_model_id} updated and saved to MinIO")
    return {"message": "Model updated successfully", "model_id": updated_model_id}
