"""
Storage service for handling file storage with support for both local filesystem and S3.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO, List, Optional, Tuple
from fastapi import UploadFile

from models import DataRegion, Project
from env import get_settings

from loguru import logger

# Get settings
settings = get_settings()

# Default upload directory
DEFAULT_UPLOAD_DIR = "uploads/files"

# S3 bucket configuration for each region
S3_CONFIG = {
    DataRegion.INDIA.value: {
        "bucket_name": settings.s3_bucket_india,
        "endpoint": settings.s3_endpoint_india,
        "region": "ap-south-1",
    },
    DataRegion.US.value: {
        "bucket_name": settings.s3_bucket_us,
        "endpoint": settings.s3_endpoint_us,
        "region": "us-east-1",
    },
}


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save_file(
        self, file: UploadFile, path: str, content_type: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Save a file to storage.

        Args:
            file: The file to save
            path: The path where to save the file
            content_type: The content type of the file

        Returns:
            Tuple containing the file path and size
        """
        pass

    @abstractmethod
    async def get_file(self, path: str) -> BinaryIO:
        """
        Get a file from storage.

        Args:
            path: The path of the file to get

        Returns:
            The file as a binary stream
        """
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from storage.

        Args:
            path: The path of the file to delete

        Returns:
            True if the file was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: The directory to list files from

        Returns:
            List of file paths
        """
        pass

    @abstractmethod
    def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """
        Get a URL for a file.

        Args:
            path: The path of the file
            expires_in: The expiration time in seconds (for pre-signed URLs)

        Returns:
            The URL for the file
        """
        pass


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem."""

    def __init__(self, base_dir: str = DEFAULT_UPLOAD_DIR):
        """
        Initialize the local storage backend.

        Args:
            base_dir: The base directory for file storage
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    async def save_file(
        self, file: UploadFile, path: str, content_type: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Save a file to local storage.

        Args:
            file: The file to save
            path: The path where to save the file
            content_type: The content type of the file (not used for local storage)

        Returns:
            Tuple containing the file path and size
        """
        full_path = os.path.join(self.base_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        size = 0
        with open(full_path, "wb") as f:
            # Read and write in chunks to avoid loading large files into memory
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                f.write(chunk)
                size += len(chunk)

        return path, size

    async def get_file(self, path: str) -> BinaryIO:
        """
        Get a file from local storage.

        Args:
            path: The path of the file to get

        Returns:
            The file as a binary stream
        """
        full_path = os.path.join(self.base_dir, path)
        return open(full_path, "rb")

    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from local storage.

        Args:
            path: The path of the file to delete

        Returns:
            True if the file was deleted, False otherwise
        """
        full_path = os.path.join(self.base_dir, path)
        try:
            os.remove(full_path)
            return True
        except FileNotFoundError:
            logger.warning(f"File not found: {full_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file {full_path}: {e}")
            return False

    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory in local storage.

        Args:
            directory: The directory to list files from

        Returns:
            List of file paths
        """
        full_dir = os.path.join(self.base_dir, directory)
        if not os.path.exists(full_dir):
            return []

        files = []
        for root, _, filenames in os.walk(full_dir):
            for filename in filenames:
                rel_path = os.path.relpath(os.path.join(root, filename), self.base_dir)
                files.append(rel_path)
        return files

    def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """
        Get a URL for a file in local storage.

        Args:
            path: The path of the file
            expires_in: The expiration time in seconds (not used for local storage)

        Returns:
            The URL for the file
        """
        # For local storage, we just return the path
        return f"/files/{path}"


class S3StorageBackend(StorageBackend):
    """
    S3 storage backend using boto3.
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint: str,
        region: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize the S3 storage backend.

        Args:
            bucket_name: The name of the S3 bucket
            endpoint: The S3 endpoint URL
            region: The AWS region
            access_key: The AWS access key ID
            secret_key: The AWS secret access key
        """
        import boto3
        from botocore.client import Config

        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.region = region
        self.access_key = access_key or settings.s3_access_key
        self.secret_key = secret_key or settings.s3_secret_key

        # Validate required credentials
        if not all([self.bucket_name, self.access_key, self.secret_key]):
            raise ValueError("Missing required S3 credentials")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4')
        )
        
        # Initialize S3 resource
        self.s3_resource = boto3.resource(
            's3',
            region_name=self.region,
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    async def save_file(
        self, file: UploadFile, path: str, content_type: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Save a file to S3 storage.

        Args:
            file: The file to save
            path: The path where to save the file
            content_type: The content type of the file

        Returns:
            Tuple containing the file path and size
        """
        import io

        try:
            # Log detailed information about the upload
            logger.info(f"Uploading {file.filename} to S3: {self.region}/{self.bucket_name}/{path}")
            
            # Read file content
            content = await file.read()
            size = len(content)
            self.s3_client.upload_fileobj(
                io.BytesIO(content),
                self.bucket_name,
                path,
                ExtraArgs={
                    'ContentType': content_type or 'application/octet-stream'
                }
            )
            
            # Generate a pre-signed URL for verification
            url = self.get_file_url(f"s3://{self.bucket_name}/{path}")
            
            logger.info(f"Upload successful!")
            logger.info(f"  S3 URI: s3://{self.bucket_name}/{path}")
            logger.info(f"  Pre-signed URL: {url}")
            logger.info(f"  Full S3 path: https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{path}")
            
            # Return the S3 path and size
            s3_path = f"s3://{self.bucket_name}/{path}"
            return s3_path, size
            
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            logger.error(f"  Bucket: {self.bucket_name}")
            logger.error(f"  Path: {path}")
            logger.error(f"  Exception details: {str(e)}")
            raise

    async def get_file(self, path: str) -> BinaryIO:
        """
        Get a file from S3 storage.

        Args:
            path: The path of the file to get

        Returns:
            The file as a binary stream
        """
        import io
        
        # Parse the S3 path
        if not path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {path}")

        parts = path[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {path}")

        bucket, key = parts
        
        try:
            # Create a BytesIO object to store the file content
            file_obj = io.BytesIO()
            
            # Download the file from S3
            self.s3_client.download_fileobj(bucket, key, file_obj)
            
            # Reset the file pointer to the beginning
            file_obj.seek(0)
            
            return file_obj
            
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise FileNotFoundError(f"File not found in S3: {path}")

    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from S3 storage.

        Args:
            path: The path of the file to delete

        Returns:
            True if the file was deleted, False otherwise
        """
        # Parse the S3 path
        if not path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {path}")

        parts = path[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {path}")

        bucket, key = parts
        
        try:
            # Delete the file from S3
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted file {key} from S3 bucket {bucket}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False

    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory in S3 storage.

        Args:
            directory: The directory to list files from

        Returns:
            List of file paths
        """
        try:
            # Ensure directory ends with a slash if it's not empty
            if directory and not directory.endswith('/'):
                directory += '/'
                
            # List objects in the bucket with the specified prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=directory
            )
            
            # Extract the keys (file paths)
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
                    
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from S3: {e}")
            return []

    def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """
        Get a pre-signed URL for a file in S3 storage.

        Args:
            path: The path of the file
            expires_in: The expiration time in seconds

        Returns:
            The pre-signed URL for the file
        """
        # Parse the S3 path
        if not path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {path}")

        parts = path[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {path}")

        bucket, key = parts
        
        try:
            # Generate a pre-signed URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket,
                    'Key': key
                },
                ExpiresIn=expires_in
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Error generating pre-signed URL: {e}")
            return f"{self.endpoint}/{bucket}/{key}"


def get_storage_backend(project: Optional[Project] = None, file_path: Optional[str] = None) -> StorageBackend:
    """
    Factory function to get the appropriate storage backend based on environment variables,
    project settings, and file path.

    Args:
        project: Optional project with storage settings
        file_path: Optional file path to determine the storage backend

    Returns:
        A storage backend instance
    """
    # If file_path is provided, use it to determine the storage backend
    if file_path:
        if file_path.startswith("s3://"):
            logger.info(f"File path starts with s3://, using S3 storage backend: {file_path}")
            # Extract the bucket and region from the S3 path
            parts = file_path[5:].split("/", 1)
            if len(parts) == 2:
                bucket, _ = parts
                # Determine the region based on the bucket name
                region = None
                if bucket == settings.s3_bucket_india:
                    region = DataRegion.INDIA.value
                elif bucket == settings.s3_bucket_us:
                    region = DataRegion.US.value
                
                if region:
                    config = S3_CONFIG.get(region)
                    if config:
                        logger.info(f"Using S3 storage backend for region {region} and bucket {bucket}")
                        return S3StorageBackend(
                            bucket_name=bucket,
                            endpoint=config["endpoint"],
                            region=config["region"],
                            access_key=settings.s3_access_key,
                            secret_key=settings.s3_secret_key,
                        )
            
            # If we couldn't determine the region or bucket, use the default S3 backend
            logger.info("Using default S3 storage backend for US region")
            config = S3_CONFIG.get(DataRegion.US.value)
            return S3StorageBackend(
                bucket_name=settings.s3_bucket_us,
                endpoint=config["endpoint"],
                region=config["region"],
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
            )
        else:
            logger.info(f"File path does not start with s3://, using local storage backend: {file_path}")
            # For local files, we need to use the correct base directory
            # If the path starts with "projects/", it's likely stored in "uploads/projects/"
            if file_path.startswith("projects/"):
                logger.info(f"File path starts with projects/, using uploads/projects as base directory")
                return LocalStorageBackend(base_dir="uploads")
            return LocalStorageBackend()
    # Log detailed information about the storage configuration
    logger.info("=== Storage Backend Selection ===")
    logger.info(f"Project provided: {bool(project)}")
    if project:
        logger.debug(f"  Project ID: {project.id}")
        logger.debug(f"  Project Name: {project.name}")
        logger.debug(f"  Project Data Region: {project.data_region}")
        logger.debug(f"  Project S3 Bucket Name: {project.s3_bucket_name}")
    
    logger.debug("Environment Settings:")
    logger.debug(f"  use_s3_storage: {settings.use_s3_storage}")
    logger.debug(f"  s3_access_key exists: {bool(settings.s3_access_key)}")
    logger.debug(f"  s3_secret_key exists: {bool(settings.s3_secret_key)}")
    logger.debug(f"  s3_bucket_us: {settings.s3_bucket_us}")
    logger.debug(f"  s3_bucket_india: {settings.s3_bucket_india}")
    
    # Check if S3 storage is enabled globally
    use_s3 = settings.use_s3_storage
    logger.info(f"S3 storage enabled globally: {use_s3}")

    # If project is provided, check its data region
    if project:
        if project.data_region == DataRegion.LOCAL.value:
            # For LOCAL data region, always use local storage regardless of global setting
            use_s3 = False
            logger.debug(f"Project {project.id} ({project.name}) has LOCAL data region, forcing local storage")
        elif project.data_region != DataRegion.LOCAL.value:
            # For non-LOCAL data regions, always use S3
            use_s3 = True
            logger.debug(f"Project has non-LOCAL data region ({project.data_region}), enabling S3 storage")

    if use_s3:
        logger.info("Using S3 storage backend")
        
        # Get the region from the project or default to US
        region = DataRegion.US.value
        if project and project.data_region:
            if project.data_region == DataRegion.INDIA.value:
                region = DataRegion.INDIA.value
                logger.info(f"Using INDIA region from project {project.id} ({project.name})")
            elif project.data_region == DataRegion.US.value:
                region = DataRegion.US.value
                logger.info(f"Using US region from project {project.id} ({project.name})")
            elif project.data_region == DataRegion.LOCAL.value:
                logger.info(f"Project {project.id} ({project.name}) has LOCAL data region, defaulting to US for S3")
            else:
                logger.warning(f"Unknown data region: {project.data_region}, defaulting to US")
        else:
            logger.info(f"No project or no data region specified, defaulting to US")
        
        logger.info(f"Selected region: {region}")

        # Get the S3 configuration for the region
        config = S3_CONFIG.get(region)
        if not config:
            logger.warning(
                f"No S3 configuration found for region {region}. "
                "Falling back to local storage."
            )
            return LocalStorageBackend()

        # Use the project's bucket name if provided, otherwise use the default
        bucket_name = (
            project.s3_bucket_name
            if project and project.s3_bucket_name
            else config["bucket_name"]
        )
        logger.info(f"Selected bucket: {bucket_name}")
        logger.info(f"S3 endpoint: {config['endpoint']}")
        logger.info(f"S3 region: {config['region']}")

        # Check if required S3 settings are available
        access_key = settings.s3_access_key
        secret_key = settings.s3_secret_key

        missing_settings = []
        if not bucket_name:
            missing_settings.append("bucket_name")
        if not access_key:
            missing_settings.append("s3_access_key")
        if not secret_key:
            missing_settings.append("s3_secret_key")

        if missing_settings:
            logger.warning(
                f"S3 storage is enabled but required settings are missing: {', '.join(missing_settings)}. "
                "Falling back to local storage."
            )
            return LocalStorageBackend()

        logger.info(f"Creating S3StorageBackend with bucket: {bucket_name}, region: {config['region']}")
        return S3StorageBackend(
            bucket_name=bucket_name,
            endpoint=config["endpoint"],
            region=config["region"],
            access_key=access_key,
            secret_key=secret_key,
        )

    # Use local storage
    logger.info("Using LocalStorageBackend")
    return LocalStorageBackend()


def get_project_storage_path(project: Project, path: str) -> str:
    """
    Get the storage path for a file in a project.

    Args:
        project: The project
        path: The file path

    Returns:
        The storage path for the file
    """
    # Create a path that includes the project ID
    return f"projects/{project.id}/{path}"
