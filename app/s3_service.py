import boto3
import os
from botocore.exceptions import NoCredentialsError
from config import S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


def create_bucket(bucket_name: str = S3_BUCKET_NAME):
    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except Exception as e:
        raise ValueError(f"Error: {e}")


def upload_to_s3(file_path: str, s3_key: str, bucket_name: str = S3_BUCKET_NAME):
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
    except FileNotFoundError:
        raise ValueError(f"File {file_path} not found")
    except NoCredentialsError:
        raise ValueError("Invalid S3 credentials")


def download_from_s3(s3_key: str, download_path: str):
    """
    Downloads a file from MinIO to the specified path.
    """
    try:
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        s3_client.download_file(S3_BUCKET_NAME, s3_key, download_path)
        print(f"Downloaded {s3_key} to {download_path}")
    except Exception as e:
        raise ValueError(f"Failed to download file: {e}")


def get_list_from_bucket(folder_name: str):
    bucket_name = S3_BUCKET_NAME
    prefix = folder_name

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    trained_models = [
        obj["Key"][obj["Key"].find("/") + 1 : obj["Key"].rfind(".")]
        for obj in response["Contents"]
    ]
    return trained_models
