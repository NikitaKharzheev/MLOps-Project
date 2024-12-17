import pytest
from moto import mock_aws
import boto3



@pytest.fixture(scope="function")
def s3_mock():
    """
    Mocked S3 bucket for testing.
    """
    with mock_aws():
        s3 = boto3.client(
            "s3",
            aws_access_key_id="fake_access_key",
            aws_secret_access_key="fake_secret_key",
            region_name="us-east-1",
        )
        # Создаём фиктивный бакет
        bucket_name = "test-bucket"
        s3.create_bucket(Bucket=bucket_name)
        yield s3, bucket_name
