"""Azure Blob Storage client — uploads captured images, returns SAS URLs."""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BlobStorageClient:
    """Azure Blob Storage client for uploading images and generating SAS URLs."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME", "captures")

        if not self.connection_string:
            raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING in .env")

        self._service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self._account_name = self._service_client.account_name
        self._account_key = self._service_client.credential.account_key
        self._ensure_container()
        logger.info(f"BlobStorageClient initialized (container: {self.container_name})")

    def _ensure_container(self) -> None:
        try:
            self._service_client.get_container_client(self.container_name).create_container()
            logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            if "ResourceExistsError" in type(e).__name__ or "already exists" in str(e).lower():
                pass  # expected on subsequent runs
            else:
                logger.warning(f"Could not ensure container exists: {e}")

    def upload_image(self, local_path: str, sas_expiry_hours: int = 2) -> str:
        """Upload a local image to blob storage and return a time-limited SAS URL."""
        blob_name = os.path.basename(local_path)
        blob_client = self._service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        expiry = datetime.now(timezone.utc) + timedelta(hours=sas_expiry_hours)
        sas_token = generate_blob_sas(
            account_name=self._account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        sas_url = f"{blob_client.url}?{sas_token}"
        logger.info(f"Uploaded '{blob_name}', SAS URL valid for {sas_expiry_hours}h")
        return sas_url
