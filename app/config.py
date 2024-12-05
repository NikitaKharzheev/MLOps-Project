import os

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio123")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-bucket")
