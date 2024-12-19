FROM python:3.9

# Устанавливаем Poetry и DVC
RUN pip install poetry dvc[all] --no-cache-dir

WORKDIR /app

# Открываем порт для FastAPI
EXPOSE 8000

# Копируем зависимости
COPY pyproject.toml poetry.lock /app/

# Копируем исходный код
COPY app /app/

# Устанавливаем зависимости проекта
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Настройка DVC remote для MinIO
RUN dvc init --no-scm \
    && dvc remote add -d minio_remote s3://mlops-bucket \
    && dvc remote modify minio_remote endpointurl http://minio:9000 \
    && dvc remote modify minio_remote access_key_id minioadmin \
    && dvc remote modify minio_remote secret_access_key minioadmin

# Запускаем FastAPI-приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



