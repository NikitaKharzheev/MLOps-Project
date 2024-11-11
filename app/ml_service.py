import uuid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class ModelManager:
    def __init__(self):
        """
        Initializes the ModelManager.

        The ModelManager holds a dictionary of models that have been trained
        and the predictions made by those models.

        :param: None
        :return: None
        """
        self.models = {}
        self.predictions = {}
        self.available_models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
        }

    def get_available_models(self):
        """Returns a list of available model types."""
        return list(self.available_models.keys())

    def train_model(self, model_type, hyperparameters, data_content, target_variable):
        """
        Trains a machine learning model of the specified type using the provided data and hyperparameters.

        This function converts the input data into a pandas DataFrame, checks for the presence of the
        target variable in the data, and trains a model based on the specified model type. The trained
        model is stored in the models dictionary with a unique model ID.

        :param model_type: The type of model to train (e.g., "logistic_regression", "random_forest").
        :param hyperparameters: A dictionary of hyperparameters for the model.
        :param data_content: The training data as a list of dictionaries or a DataFrame-compatible structure.
        :param target_variable: The name of the target variable in the data.

        :raises ValueError: If the target variable is not found in the data.

        :return: A unique identifier for the trained model.
        """
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
        """
        Returns a list of unique IDs for all trained models.
        """

        return list(self.models.keys())

    def predict(self, model_id, data_content):
        """
        Makes predictions using the specified trained model.

        This function takes a model ID and the input data to make predictions and
        returns a list of predictions. The input data should be a list of
        dictionaries or a DataFrame-compatible structure.

        :param model_id: The ID of the trained model to use for making predictions.
        :param data_content: The input data to make predictions.

        :raises ValueError: If the target variable is not found in the data.

        :return: A list of predictions or None if an error occurs.
        """

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

    def update_model(
        self, model_id, data_content, hyperparameters=None, target_variable=None
    ):
        """
        Updates a trained model with new data and/or hyperparameters.

        This function takes a model ID, new data to update the model with, and
        optional hyperparameters and target variable. It will update the model
        with the new data and hyperparameters, and return True if the update is
        successful.

        If the target variable is not provided, the default target variable set
        when the model was trained is used.

        If the hyperparameters are not provided, the existing hyperparameters of
        the model are used.

        :param model_id: The ID of the model to update.
        :param data_content: The new data to update the model with.
        :param hyperparameters: Optional hyperparameters to update the model with.
        :param target_variable: Optional target variable to use for updating the model.

        :raises ValueError: If the target variable is not found in the data.

        :return: True if the update is successful, False otherwise.
        """
        model = self.models.get(model_id)
        if model is None:
            return False

        df = pd.DataFrame(data_content)
        if target_variable and target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")

        X = (
            df.drop(columns=[target_variable])
            if target_variable
            else df.drop(columns=[self.default_target_variable])
        )
        y = df[target_variable] if target_variable else df[self.default_target_variable]

        model_cls = type(model)
        model = model_cls(**hyperparameters) if hyperparameters else model
        model.fit(X, y)

        self.models[model_id] = model
        return True

    def delete_model(self, model_id):
        """
        Deletes a trained model from the ModelManager.

        This function removes the model associated with the provided model ID
        from the models dictionary.

        :param model_id: The ID of the model to be deleted.

        :return: True if the model was successfully deleted, False if the model ID was not found.
        """
        return self.models.pop(model_id, None) is not None

    def get_predictions(self, model_id):
        """
        Retrieves the saved predictions for a specific model.

        This function fetches the list of predictions associated with the given
        model ID from the predictions dictionary.

        :param model_id: The ID of the model for which to retrieve predictions.

        :return: A list of predictions if available, or None if the model ID is not found.
        """
        return self.predictions.get(model_id, None)
