import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.s3_service import upload_to_s3
import os

client = TestClient(app)


def test_status_endpoint():
    """
    Test the /status endpoint.
    """
    response = client.get("/status")
    assert response.status_code == 200
    assert "Service is running" in response.json()["status"]


def test_upload_to_s3(s3_mock):
    """
    Test upload_to_s3 using a mocked S3 bucket.
    """
    s3, bucket_name = s3_mock
    os.environ["AWS_STORAGE_BUCKET_NAME"] = bucket_name

    # Загружаем тестовый файл
    test_file_content = b"test content"
    test_file_path = "/tmp/test_file.txt"
    s3_key = "test_file.txt"

    with open(test_file_path, "wb") as f:
        f.write(test_file_content)

    # Тестируем функцию upload_to_s3
    upload_to_s3(test_file_path, s3_key)

    # Проверяем, что файл появился в S3
    response = s3.get_object(Bucket=bucket_name, Key=s3_key)
    assert response["Body"].read() == test_file_content

    # Удаляем временный файл
    os.remove(test_file_path)
