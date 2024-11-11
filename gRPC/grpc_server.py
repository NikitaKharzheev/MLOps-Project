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
    filename="py_log.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        self.model_manager = ModelManager()
        self.uploaded_data_storage = {}
        logger.info("ModelServiceServicer initialized")

    def UploadData(self, request, context):
        try:
            data_content = json.loads(request.data)
            self.uploaded_data_storage["last_uploaded_data"] = data_content
            logger.info("Data uploaded successfully")
            return model_service_pb2.UploadDataResponse(message="Data uploaded successfully")
        except json.JSONDecodeError:
            logger.error("Failed to upload data: Invalid JSON format")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid JSON format")
            return model_service_pb2.UploadDataResponse()

    def TrainModel(self, request, context):
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
            target_variable=request.target_variable
        )
        logger.info(f"Training started for model ID: {model_id}")
        return model_service_pb2.TrainModelResponse(message="Training started", model_id=model_id)

    def ListAvailableModels(self, request, context):
        model_types = self.model_manager.get_available_models()
        logger.info("Listed available models")
        return model_service_pb2.ListAvailableModelsResponse(model_types=model_types)

    def Predict(self, request, context):
        model = self.model_manager.models.get(request.model_id)
        if model is None:
            logger.error(f"Predict request failed: Model {request.model_id} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found or not trained")
            return model_service_pb2.PredictResponse()

        try:
            data_content = json.loads(request.data)
            df = pd.DataFrame(data_content) if isinstance(data_content, list) else pd.DataFrame([data_content])
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
        result = self.model_manager.delete_model(request.model_id)
        if result:
            logger.info(f"Model {request.model_id} deleted successfully")
            return model_service_pb2.DeleteModelResponse(message="Model deleted")
        else:
            logger.error(f"DeleteModel request failed: Model {request.model_id} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return model_service_pb2.DeleteModelResponse()

    def Status(self, request, context):
        logger.info("Received Status request")
        print("Received Status request")
        return model_service_pb2.StatusResponse(status="Service is running")

    def ListTrainedModels(self, request, context):
        trained_models = self.model_manager.get_trained_models()
        logger.info("Listed trained models")
        return model_service_pb2.ListTrainedModelsResponse(model_ids=trained_models)
    def GetPredictions(self, request, context):
        predictions = self.model_manager.get_predictions(request.model_id)
        if predictions is None:
            logger.error(f"GetPredictions request failed: Model {request.model_id} not found or no predictions available")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found or no predictions available")
            return model_service_pb2.GetPredictionsResponse()
        
        logger.info(f"Predictions retrieved for model ID: {request.model_id}")
        return model_service_pb2.GetPredictionsResponse(predictions=predictions)

    def UpdateModel(self, request, context):
        data_content = self.uploaded_data_storage.get("last_uploaded_data")
        result = self.model_manager.update_model(
            model_id=request.model_id,
            data_content=data_content,
            hyperparameters=dict(request.hyperparameters) if request.hyperparameters else None,
            target_variable=request.target_variable
        )
        if result:
            logger.info(f"Model {request.model_id} updated successfully")
            return model_service_pb2.UpdateModelResponse(message="Model updated successfully")
        else:
            logger.error(f"UpdateModel request failed: Model {request.model_id} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return model_service_pb2.UpdateModelResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()