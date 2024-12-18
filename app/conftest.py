import pytest
from moto import mock_aws
import boto3
from config import S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY


@pytest.fixture(scope="function")
def s3_mock():
    """
    Mocked S3 bucket for testing.
    """
    with mock_aws():
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

        # Создаём фиктивный бакет
        bucket_name = "test-bucket"
        try:
            s3.create_bucket(Bucket=bucket_name)
        finally:
            yield s3, bucket_name
