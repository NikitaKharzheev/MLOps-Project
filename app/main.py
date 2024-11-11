from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ml_service import ModelManager
import psutil
import json
import logging

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    filename="py_log.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
)
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
    """
    Uploads a JSON file containing data for model training/prediction.

    The provided file should be a JSON, and the contents will be stored in memory
    for subsequent requests. The first 5 elements of the uploaded data will be
    returned as a preview.

    :param data: The JSON file to upload. The file should be a JSON and should
        contain the data for model training/prediction.
    :return: A JSON response containing a success message, as well as a preview
        of the first 5 elements of the uploaded data.
    """

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
    """
    Trains a model on the uploaded data.

    :param train_request: A JSON payload containing the model type, hyperparameters, and target variable.
    :return: A JSON response containing a success message and the model ID.
    """
    logger.info("Received model training request")
    data_content = uploaded_data_storage.get("last_uploaded_data")
    if not data_content:
        logger.error("No data uploaded before training request")
        raise HTTPException(
            status_code=400,
            detail="No data uploaded. Please upload data first using /upload-data",
        )

    if train_request.model_type not in model_manager.get_available_models():
        logger.error(f"Unsupported model type: {train_request.model_type}")
        raise HTTPException(status_code=400, detail="Model type not supported")

    model_id = model_manager.train_model(
        model_type=train_request.model_type,
        hyperparameters=train_request.hyperparameters,
        data_content=data_content,
        target_variable=train_request.target_variable,
    )
    logger.info(f"Training started for model {model_id}")
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
    Retrieves a prediction for a given model and input data.

    This endpoint receives a JSON file containing the input data and returns a
    JSON response containing the prediction. The input data should be a JSON
    object with the same structure as the data that was used to train the model.

    The prediction is generated using the model with the given ID. If the model
    ID is invalid or the model has not been trained, a 404 error is returned.

    :param model_id: The ID of the model to use for prediction.
    :param data: The input data to use for prediction.
    :return: A JSON response containing the prediction.
    """
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
    """
    Deletes a model with the given ID.

    This endpoint deletes the model with the given ID. If the model ID is
    invalid or the model has not been trained, a 404 error is returned.

    :param model_id: The ID of the model to delete.
    :return: A JSON response indicating whether the model was deleted
        successfully.
    """
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
    """
    Retrieves information about the service status.

    This endpoint returns a JSON response containing information about the
    service status. The response contains the following fields:

    - status: A string indicating the status of the service.
    - available_memory: A string indicating the amount of available memory in
      megabytes.
    - used_memory: A string indicating the amount of used memory in megabytes.

    :return: A JSON response containing service status information.
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
    """
    Retrieves a list of all trained models.

    This endpoint returns a JSON response containing a list of all trained
    models. Each model is represented as a string containing the model ID.

    :return: A JSON response containing a list of trained models.
    """
    logger.info("Listing trained models")
    trained_models = model_manager.get_trained_models()
    return {"trained_models": trained_models}


@app.get("/prediction/{model_id}")
async def get_predictions(model_id: str):
    """
    Retrieves saved predictions for a specified model.

    This endpoint fetches the predictions that have been previously generated
    and saved for the model identified by the given model_id. If no predictions
    are found, or if the model does not exist, a 404 error is returned.

    :param model_id: The ID of the model for which to retrieve predictions.
    :return: A JSON response containing the saved predictions.
    """
    logger.info(f"Fetching saved predictions for model {model_id}")
    saved_predictions = model_manager.get_predictions(model_id)
    if saved_predictions is None:
        logger.error(f"Predictions not found for model {model_id}")
        raise HTTPException(
            status_code=404, detail="Model not found or no predictions available"
        )

    return {"predictions": saved_predictions}


@app.put("/update-model/{model_id}")
async def update_model(model_id: str, update_request: UpdateRequest):
    """
    Updates an existing model with new data and hyperparameters.

    This endpoint updates the model identified by the given model_id using the
    newly uploaded data and the specified hyperparameters and target variable.
    If the model is successfully updated, a success message is returned. If the
    model ID is invalid or the model does not exist, a 404 error is returned.

    :param model_id: The ID of the model to update.
    :param update_request: A JSON payload containing the hyperparameters and
                           target variable for the update.
    :return: A JSON response indicating whether the model was updated
             successfully.
    """
    logger.info(f"Received update request for model {model_id}")
    data_content = uploaded_data_storage.get("last_uploaded_data")
    result = model_manager.update_model(
        model_id=model_id,
        data_content=data_content,
        hyperparameters=update_request.hyperparameters,
        target_variable=update_request.target_variable,
    )
    if result:
        logger.info(f"Model {model_id} updated successfully")
        return {"message": "Model updated successfully"}
    else:
        logger.error(f"Model {model_id} not found for update")
        raise HTTPException(status_code=404, detail="Model not found")
