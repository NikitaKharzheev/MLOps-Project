import uuid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ModelManager:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.available_models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier
        }

    def get_available_models(self):
        return list(self.available_models.keys())

    def train_model(self, model_type, hyperparameters, data_content, target_variable):
        df = pd.DataFrame(data_content)
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")

        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        model_cls = self.available_models.get(model_type)
        model = model_cls(**hyperparameters)
        model.fit(X, y)
        
        model_id = str(uuid.uuid4())
        self.models[model_id] = model
        self.predictions[model_id] = []
        return model_id

    def get_trained_models(self):
        return list(self.models.keys())

    def predict(self, model_id, data_content):
        model = self.models.get(model_id)
        if model is None:
            print(f"No model found for model_id: {model_id}")
            return None

        print(f"Data received in ModelManager.predict: {data_content}")

        try:
            df = pd.DataFrame(data_content)
        except Exception as e:
            print(f"Error converting data to DataFrame: {e}")
            return None

        try:
            predictions = model.predict(df).tolist()
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None

        print(f"Predictions: {predictions}")

        if model_id not in self.predictions:
            self.predictions[model_id] = []
        self.predictions[model_id].extend(predictions)
        return predictions

    def update_model(self, model_id, data_content, hyperparameters=None, target_variable=None):
        model = self.models.get(model_id)
        if model is None:
            return False
        
        df = pd.DataFrame(data_content)
        if target_variable and target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")

        X = df.drop(columns=[target_variable]) if target_variable else df.drop(columns=[self.default_target_variable])
        y = df[target_variable] if target_variable else df[self.default_target_variable]

        model_cls = type(model)
        model = model_cls(**hyperparameters) if hyperparameters else model
        model.fit(X, y)
        
        self.models[model_id] = model
        return True

    def delete_model(self, model_id):
        return self.models.pop(model_id, None) is not None
    
    def get_predictions(self, model_id):
        return self.predictions.get(model_id, None)