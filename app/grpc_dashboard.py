import streamlit as st
import grpc
import json
import model_service_pb2
import model_service_pb2_grpc

GRPC_SERVER_ADDRESS = "localhost:50051"

channel = grpc.insecure_channel(GRPC_SERVER_ADDRESS)
stub = model_service_pb2_grpc.ModelServiceStub(channel)

st.title("Панель управления ML моделями (gRPC)")

st.sidebar.title("Навигация")
options = [
    "Загрузить данные",
    "Обучить модель",
    "Сделать предсказание",
    "Просмотр моделей",
    "Просмотр предсказаний",
    "Обновить модель",
    "Удалить модель",
    "Проверка статуса сервиса",
]
choice = st.sidebar.selectbox("Выберите действие", options)

if choice == "Загрузить данные":
    st.subheader("Загрузите набор данных (в формате JSON)")
    uploaded_file = st.file_uploader("Выберите файл JSON", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            data_json = json.dumps(data)
            request = model_service_pb2.UploadDataRequest(data=data_json)
            response = stub.UploadData(request)
            st.write(response.message)
        except json.JSONDecodeError:
            st.error("Недопустимый формат JSON файла.")
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")

elif choice == "Обучить модель":
    st.subheader("Обучение модели")
    model_type = st.selectbox(
        "Выберите тип модели", ["logistic_regression", "random_forest"]
    )
    target_variable = st.text_input("Введите целевую переменную")
    hyperparameters_input = st.text_area("Введите гиперпараметры")
    if st.button("Обучить"):
        try:
            hyperparameters = (
                json.loads(hyperparameters_input) if hyperparameters_input else {}
            )
            request = model_service_pb2.TrainModelRequest(
                model_type=model_type,
                hyperparameters=hyperparameters,
                target_variable=target_variable,
            )
            response = stub.TrainModel(request)
            st.success(f"Обучение запущено, ID модели: {response.model_id}")
        except json.JSONDecodeError:
            st.error("Недопустимый формат JSON для гиперпараметров.")
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")

elif choice == "Сделать предсказание":
    st.subheader("Сделать предсказание")
    model_id = st.text_input("Введите ID модели")
    uploaded_file = st.file_uploader(
        "Выберите файл JSON для предсказания", type=["json"]
    )
    if uploaded_file is not None and st.button("Предсказать"):
        try:
            data = json.load(uploaded_file)
            data_json = json.dumps(data)
            request = model_service_pb2.PredictRequest(
                model_id=model_id, data=data_json
            )
            response = stub.Predict(request)
            st.write("Предсказания:", response.prediction)
        except json.JSONDecodeError:
            st.error("Недопустимый формат JSON файла.")
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")

elif choice == "Просмотр моделей":
    st.subheader("Доступные модели")
    try:
        response = stub.ListAvailableModels(model_service_pb2.Empty())
        st.write("Доступные типы моделей:", response.model_types)
    except grpc.RpcError as e:
        st.error(f"Ошибка: {e.details()}")

    st.subheader("Обученные модели")
    try:
        response = stub.ListTrainedModels(model_service_pb2.Empty())
        st.write("Обученные модели:", response.model_ids)
    except grpc.RpcError as e:
        st.error(f"Ошибка: {e.details()}")

elif choice == "Просмотр предсказаний":
    st.subheader("Просмотр предсказаний для модели")
    model_id = st.text_input("Введите ID модели для отображения предсказаний")
    if st.button("Получить предсказания"):
        try:
            request = model_service_pb2.GetPredictionsRequest(model_id=model_id)
            response = stub.GetPredictions(request)
            st.write("Предсказания:", response.predictions)
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")


elif choice == "Обновить модель":
    st.subheader("Обновление модели")

    model_id = st.text_input("Введите ID модели для обновления")
    target_variable = st.text_input("Введите новую целевую переменную")
    hyperparameters_input = st.text_area("Введите новые гиперпараметры")

    if st.button("Обновить модель"):
        try:
            hyperparameters = (
                json.loads(hyperparameters_input) if hyperparameters_input else {}
            )
            request = model_service_pb2.UpdateModelRequest(
                model_id=model_id,
                target_variable=target_variable,
                hyperparameters=hyperparameters,
            )
            response = stub.UpdateModel(request)
            st.success(response.message)
        except json.JSONDecodeError:
            st.error("Недопустимый формат JSON для гиперпараметров.")
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")

elif choice == "Удалить модель":
    st.subheader("Удаление модели")

    model_id = st.text_input("Введите ID модели для удаления")
    if st.button("Удалить"):
        try:
            request = model_service_pb2.DeleteModelRequest(model_id=model_id)
            response = stub.DeleteModel(request)
            st.success(response.message)
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")


elif choice == "Проверка статуса сервиса":
    st.subheader("Проверка статуса сервиса")
    if st.button("Проверить статус"):
        try:
            response = stub.Status(model_service_pb2.Empty())
            st.write("Статус сервиса:", response.status)
        except grpc.RpcError as e:
            st.error(f"Ошибка: {e.details()}")
