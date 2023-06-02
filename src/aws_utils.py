from pathlib import Path
import logging
import boto3


# Set up the logger
logger = logging.getLogger(__name__)


def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    """
    This function downloads a file from an S3 bucket.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    object_key (str): The key of the object in the S3 bucket.
    local_file_path (Path): The local path where the file should be downloaded.

    Returns: None
    """
    # Create the S3 client
    s3 = boto3.client("s3")

    logger.info(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    
    try:
        # Attempt to download the file
        s3.download_file(bucket_name, object_key, str(local_file_path))

        logger.info(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise