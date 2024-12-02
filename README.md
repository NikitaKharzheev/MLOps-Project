# MLOps-Project
## Команда: 
1. Муханько Артём
2. Самбуев Михаил
3. Харжеев Никита

## Описание проекта:

Этот проект предоставляет API для управления обучением моделей машинного обучения, включая:
- Загрузку данных для обучения.
- Настройку гиперпараметров и выбор целевой переменной.
- Обучение модели и получение предсказаний.
- Удаление и повторное обучение моделей.
- Проверку статуса сервиса с отображением доступной оперативной памяти.

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/NikitaKharzheev/MLOps-Project.git
```

### 2. Установка зависимостей с использованием Poetry

Установите зависимости, указанные в pyproject.toml, с помощью команды:
```bash
poetry install
```

### 3. Переход в директорию проекта

Переходите в директорию проекта с помощью команды:
```bash
cd MLOps-Project/app
```

### 4. Запуск сервера FastAPI

Для запуска FastAPI используйте следующую команду:
```bash
uvicorn main:app --reload
```

После запуска сервер будет доступен по адресу http://127.0.0.1:8000.

### 5. Работа с API через Swagger UI

Откроите браузер и перейдите по адресу http://127.0.0.1:8000/docs. Вы можете использовать Swagger UI для взаимодействия с API.

### 6. Эндпоинты API

- Загрузка данных для обучения:

    POST /upload-data  
    Параметры: JSON-файл с данными (используйте файл iris.json)  
    Описание: Загружает JSON-файл, который будет использоваться для обучения модели.

- Обучение модели:

    POST /train  
    Параметры в теле запроса:  
    model_type: тип модели (например, logistic_regression, random_forest).  
    hyperparameters: гиперпараметры для модели.  
    target_variable: имя целевой переменной в загруженных данных ("Species" для файла iris.json).  
    Описание: Обучает модель с указанными параметрами на загруженных данных.

- Получение предсказаний:

    POST /predict/{model_id}  
    Параметры:  
    model_id: ID обученной модели.  
    JSON-файл с данными для предсказания (используйте файл iris_test.json).  
    Описание: Возвращает предсказания для загруженного набора данных.

- Повторное обучение модели:

    UPADTE /update-model/{model_id}  
    Параметры:  
    model_id: ID обученной модели.  
    JSON-файл с данными для повторного обучения (файл iris_test.json).  
    Описание: Повторно обучает модель с указанными параметрами на загруженных данных.

- Получение списка доступных для обучения моделей:

    GET /models  
    Описание: Возвращает список доступных для обучения моделей.  

- Удаление модели:

    DELETE /models/{model_id}  
    Параметры: ID модели.  
    Описание: Удаляет модель по ID.

- Проверка статуса сервиса:

    GET /status  
    Описание: Возвращает статус сервиса и объём доступной оперативной памяти.

- Получения списка обученных моделей:

    GET /trained-models  
    Описание: Возвращает список обученных моделей.

- Получение предсказаний для модели:

    GET /prediction/{model_id}  
    Параметры:  
    model_id: ID обученной модели.  
    Описание: Возвращает предсказания для модели.

### 7. Запуск Streamlit для дашборда

Для запуска Streamlit дашборда выполните следующую команду:

```bash
streamlit run dashboard.py
```

После запуска Streamlit дашборда будет доступен по адресу http://127.0.0.1:8501.

### 8. Запуск gRPC-сервера

Для запуска gRPC-сервера выполните следующую команду:

```bash
python grpc_server.py
```

### 9. Запуск Streamlit для дашборда с gRPC-сервером

Для запуска Streamlit дашборда с gRPC-сервером выполните следующую команду:

```bash
streamlit run grpc_dashboard.py
```




