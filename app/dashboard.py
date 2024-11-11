import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"  

st.title("Панель управления ML моделями")

st.sidebar.title("Навигация")
options = ["Загрузить данные", "Обучить модель", "Сделать предсказание", "Просмотр моделей", "Просмотр предсказаний", "Обновить модель", "Удалить модель"]
choice = st.sidebar.selectbox("Выберите действие", options)

if choice == "Загрузить данные":
    st.subheader("Загрузите набор данных (в формате JSON)")

    uploaded_file = st.file_uploader("Выберите файл JSON", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)  
            response = requests.post(f"{API_URL}/upload-data", files={"data": ("file.json", json.dumps(data), "application/json")})
            st.write(response.json())
        except json.JSONDecodeError:
            st.error("Недопустимый формат JSON файла.")

elif choice == "Обучить модель":
    st.subheader("Обучение модели")

    model_type = st.selectbox("Выберите тип модели", ["logistic_regression", "random_forest"])
    target_variable = st.text_input("Введите целевую переменную")
    hyperparameters_input = st.text_area("Введите гиперпараметры (в формате JSON)")

    if st.button("Обучить"):
        hyperparameters = json.loads(hyperparameters_input) if hyperparameters_input else {}
        train_data = {
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "target_variable": target_variable,
        }
        response = requests.post(f"{API_URL}/train", json=train_data)
        st.write(response.json())

elif choice == "Сделать предсказание":
    st.subheader("Сделать предсказание")

    model_id = st.text_input("Введите ID модели")
    uploaded_file = st.file_uploader("Выберите файл JSON для предсказания", type=["json"])
    if uploaded_file is not None and st.button("Предсказать"):
        data = uploaded_file.read()
        response = requests.post(f"{API_URL}/predict/{model_id}", files={"data": data})
        st.write(response.json())

elif choice == "Просмотр моделей":
    st.subheader("Доступные модели")
    response = requests.get(f"{API_URL}/models")
    models = response.json()
    st.write(models)

    st.subheader("Обученные модели")
    response = requests.get(f"{API_URL}/trained-models")
    trained_models = response.json().get("trained_models", [])
    st.write(trained_models)

elif choice == "Просмотр предсказаний":
    st.subheader("Просмотр предсказаний для модели")

    model_id = st.text_input("Введите ID модели для отображения предсказаний")
    if st.button("Получить предсказания"):
        response = requests.get(f"{API_URL}/prediction/{model_id}")
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            st.write(predictions)
        else:
            st.error(f"Ошибка: {response.json().get('detail', 'Не удалось получить предсказания')}")

elif choice == "Обновить модель":
    st.subheader("Обновление модели")

    model_id = st.text_input("Введите ID модели для обновления")
    target_variable = st.text_input("Введите новую целевую переменную (необязательно)")
    hyperparameters_input = st.text_area("Введите новые гиперпараметры (в формате JSON, необязательно)")

    if st.button("Обновить модель"):
        hyperparameters = json.loads(hyperparameters_input) if hyperparameters_input else {}
        update_data = {
            "target_variable": target_variable,
            "hyperparameters": hyperparameters
        }
        response = requests.put(f"{API_URL}/update-model/{model_id}", json=update_data)
        st.write(response.json())

elif choice == "Удалить модель":
    st.subheader("Удаление модели")

    model_id = st.text_input("Введите ID модели для удаления")
    if st.button("Удалить"):
        response = requests.delete(f"{API_URL}/models/{model_id}")
        st.write(response.json())


