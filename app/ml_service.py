import uuid
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class ModelManager:
    def __init__(self):
        self.available_models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
        }

    def get_available_models(self):
        """Returns a list of available model types."""
        return list(self.available_models.keys())

    def train_model(self, model_type, hyperparameters, data_content, target_variable):
        """Trains a machine learning model."""
        df = pd.DataFrame(data_content)
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        model_cls = self.available_models[model_type]
        model = model_cls(**hyperparameters)
        model.fit(X, y)
        model_id = str(uuid.uuid4())
        return model_id

    def save_model(self, model_id, path):
        """Saves the model to a file."""
        joblib.dump(self.available_models.get(model_id), path)

    def predict(self, model_id, data_content, model_path):
        """Loads the model and makes predictions."""
        model = joblib.load(model_path)
        df = pd.DataFrame(data_content)
        return model.predict(df).tolist()
