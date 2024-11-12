import grpc
from concurrent import futures
import model_service_pb2
import model_service_pb2_grpc
from ml_service import ModelManager
import json
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="grpc_log.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        """
        Initializes the ModelServiceServicer.

        Attributes:
            model_manager (ModelManager): An instance of the ModelManager class
                to manage the machine learning models.
            uploaded_data_storage (dict): A dictionary to store the last uploaded
                data. The key is "last_uploaded_data" and the value is a JSON
                loaded dictionary containing the uploaded data.
        """
        self.model_manager = ModelManager()
        self.uploaded_data_storage = {}
        logger.info("ModelServiceServicer initialized")

    def UploadData(self, request, context):
        """Uploads data to the model service.

        The data should be a JSON formatted string.

        Args:
            request: A UploadDataRequest message containing the data to upload.
            context: The gRPC context object.

        Returns:
            An UploadDataResponse message containing a success message if the
            data is uploaded successfully.

        Raises:
            grpc.StatusCode.INVALID_ARGUMENT: If the provided data is not a valid
                JSON string.
        """
        try:
            data_content = json.loads(request.data)
            self.uploaded_data_storage["last_uploaded_data"] = data_content
            logger.info("Data uploaded successfully")
            return model_service_pb2.UploadDataResponse(
                message="Data uploaded successfully"
            )
        except json.JSONDecodeError:
            logger.error("Failed to upload data: Invalid JSON format")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid JSON format")
            return model_service_pb2.UploadDataResponse()

    def TrainModel(self, request, context):
        """Trains a machine learning model on the last uploaded data.

        Args:
            request: A TrainModelRequest message containing the model type,
                hyperparameters, and target variable.
            context: The gRPC context object.

        Returns:
            A TrainModelResponse message containing a success message and the
            model ID if the training is successful.

        Raises:
            grpc.StatusCode.FAILED_PRECONDITION: If no data has been uploaded yet.
        """
        data_content = self.uploaded_data_storage.get("last_uploaded_data")
        if not data_content:
            logger.warning("TrainModel request failed: No data uploaded")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No data uploaded. Please upload data first")
            return model_service_pb2.TrainModelResponse()

        model_id = self.model_manager.train_model(
            model_type=request.model_type,
            hyperparameters=dict(request.hyperparameters),
            data_content=data_content,
            target_variable=request.target_variable,
        )
        logger.info(f"Training started for model ID: {model_id}")
        return model_service_pb2.TrainModelResponse(
            message="Training started", model_id=model_id
        )

    def ListAvailableModels(self, request, context):
        """Returns a list of available machine learning models that can be trained.

        Args:
            request: An Empty message.
            context: The gRPC context object.

        Returns:
            A ListAvailableModelsResponse message containing a list of model types.

        Raises:
            None
        """
        model_types = self.model_manager.get_available_models()
        logger.info("Listed available models")
        return model_service_pb2.ListAvailableModelsResponse(model_types=model_types)

    def Predict(self, request, context):
        """Makes predictions on a given model for the given input data.

        Args:
            request: A PredictRequest message containing the model ID and input data.
            context: The gRPC context object.

        Returns:
            A PredictResponse message containing the predictions.

        Raises:
            grpc.StatusCode.NOT_FOUND: If the model ID is not found in the model manager.
            grpc.StatusCode.INTERNAL: If an error occurs while making predictions.
        """
        model = self.model_manager.models.get(request.model_id)
        if model is None:
            logger.error(f"Predict request failed: Model {request.model_id} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found or not trained")
            return model_service_pb2.PredictResponse()

        try:
            data_content = json.loads(request.data)
            df = (
                pd.DataFrame(data_content)
                if isinstance(data_content, list)
                else pd.DataFrame([data_content])
            )
            predictions = model.predict(df).tolist()
            if request.model_id not in self.model_manager.predictions:
                self.model_manager.predictions[request.model_id] = []
            self.model_manager.predictions[request.model_id].extend(predictions)
            logger.info(f"Predictions made for model ID: {request.model_id}")
            return model_service_pb2.PredictResponse(prediction=predictions)
        except Exception as e:
            logger.error(f"Prediction failed for model ID: {request.model_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction failed: {e}")
            return model_service_pb2.PredictResponse()

    def DeleteModel(self, request, context):
        """Deletes a model with the given ID.

        Args:
            request: A DeleteModelRequest message containing the model ID to delete.
            context: The gRPC context object.

        Returns:
            A DeleteModelResponse message containing a success message if the model
            was deleted successfully.

        Raises:
            grpc.StatusCode.NOT_FOUND: If the model ID is not found in the model manager.
        """
        result = self.model_manager.delete_model(request.model_id)
        if result:
            logger.info(f"Model {request.model_id} deleted successfully")
            return model_service_pb2.DeleteModelResponse(message="Model deleted")
        else:
            logger.error(
                f"DeleteModel request failed: Model {request.model_id} not found"
            )
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return model_service_pb2.DeleteModelResponse()

    def Status(self, request, context):
        """Returns a message indicating the status of the model service.

        Args:
            request: An empty message.
            context: The gRPC context object.

        Returns:
            A StatusResponse message containing the status of the service.
        """
        logger.info("Received Status request")
        print("Received Status request")
        return model_service_pb2.StatusResponse(status="Service is running")

    def ListTrainedModels(self, request, context):
        """Retrieves a list of trained models.

        Args:
            request: An Empty message.
            context: The gRPC context object.

        Returns:
            A ListTrainedModelsResponse message containing a list of trained model IDs.

        Raises:
            None
        """
        trained_models = self.model_manager.get_trained_models()
        logger.info("Listed trained models")
        return model_service_pb2.ListTrainedModelsResponse(model_ids=trained_models)

    def GetPredictions(self, request, context):
        """Retrieves the predictions for a given model ID.

        Args:
            request: A GetPredictionsRequest message containing the model ID.
            context: The gRPC context object.

        Returns:
            A GetPredictionsResponse message containing the predictions or an empty message
            if the model ID is not found or no predictions are available.

        Raises:
            grpc.StatusCode.NOT_FOUND: If the model ID is not found in the model manager.
        """
        predictions = self.model_manager.get_predictions(request.model_id)
        if predictions is None:
            logger.error(
                f"GetPredictions request failed: Model {request.model_id} not found or no predictions available"
            )
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found or no predictions available")
            return model_service_pb2.GetPredictionsResponse()

        logger.info(f"Predictions retrieved for model ID: {request.model_id}")
        return model_service_pb2.GetPredictionsResponse(predictions=predictions)

    def UpdateModel(self, request, context):
        """
        Updates an existing machine learning model with new data and hyperparameters.

        This function handles an update request for a model identified by the provided
        model ID. It retrieves the last uploaded data and updates the model using the
        given hyperparameters and target variable.

        Args:
            request: An UpdateModelRequest message containing the model ID, hyperparameters,
                    and target variable for the update.
            context: The gRPC context object.

        Returns:
            An UpdateModelResponse message containing a success message if the model
            was updated successfully.

        Raises:
            grpc.StatusCode.NOT_FOUND: If the model ID is not found in the model manager.
        """
        data_content = self.uploaded_data_storage.get("last_uploaded_data")
        result = self.model_manager.update_model(
            model_id=request.model_id,
            data_content=data_content,
            hyperparameters=(
                dict(request.hyperparameters) if request.hyperparameters else None
            ),
            target_variable=request.target_variable,
        )
        if result:
            logger.info(f"Model {request.model_id} updated successfully")
            return model_service_pb2.UpdateModelResponse(
                message="Model updated successfully"
            )
        else:
            logger.error(
                f"UpdateModel request failed: Model {request.model_id} not found"
            )
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return model_service_pb2.UpdateModelResponse()


def serve():
    """
    Initializes and starts the gRPC server to serve the ModelService.

    The server is configured to handle incoming gRPC requests using a thread
    pool executor with a maximum of 10 workers. It registers the
    ModelServiceServicer to the server and binds it to the address [::]:50051.

    The server logs a message when it starts and runs indefinitely until
    terminated.

    Raises:
        Exception: If the server fails to start or encounters an error during
        execution.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    logger.info("Server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
