import boto3
from botocore.exceptions import NoCredentialsError
from config import S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def upload_to_s3(file_path: str, s3_key: str):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    except FileNotFoundError:
        raise ValueError(f"File {file_path} not found")
    except NoCredentialsError:
        raise ValueError("Invalid S3 credentials")

def download_from_s3(s3_key: str, download_path: str):
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, download_path)
    except Exception as e:
        raise ValueError(f"Failed to download file: {e}")
