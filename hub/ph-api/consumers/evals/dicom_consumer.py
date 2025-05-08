import os
import sys
import asyncio
import traceback
import tempfile
from datetime import datetime, UTC
from loguru import logger
from sqlalchemy import select

from db import SessionLocal

from models import File as FileModel, ProcessingStatus, Project
from services.storage import get_storage_backend

# Add the model directory to the Python path
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model"
)
sys.path.append(models_dir)

# Directory where files are stored
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "files"
)

# Import the EchoPrime model functions
try:
    from model.echoprime.model_processor import (
        process_dicom_batch,
        DicomProcessingError,
    )

    logger.info("Successfully imported EchoPrime model")
except ImportError:
    logger.error(
        f"Failed to import EchoPrime model. Make sure it's available at {models_dir}/echoprime/"
    )

    # Define a placeholder for the model processo
    class DicomProcessingError(Exception):
        pass

    def process_dicom_batch(input_dir):
        """Placeholder for the actual model processor"""
        logger.warning("Using placeholder for EchoPrime model processor")
        return {
            "report": "This is a placeholder report. The actual model could not be loaded.",
            "metrics": {
                "ejection_fraction": 50.0,
                "rv_systolic_function_depressed": 0.0,
                "right_ventricle_dilation": 0.0,
            },
        }


async def process_dicom_file(file_id: int, user_id: int):
    """
    Process a DICOM file using the EchoPrime model.

    Args:
        file_id: The ID of the DICOM file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the File record
        file_stmt = select(FileModel).where(
            FileModel.id == file_id,
            FileModel.user_id == user_id,
            FileModel.file_type == "dicom",
        )
        file_result = db.execute(file_stmt)
        file = file_result.scalar_one_or_none()

        if not file:
            logger.error(f"DICOM file not found: file_id={file_id}, user_id={user_id}")
            return

        # Update status to processing
        file.processing_status = ProcessingStatus.PROCESSING
        db.commit()

        try:
            # Get the project if file is associated with one
            project = None
            if file.project_id:
                project_stmt = select(Project).where(Project.id == file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()

            # Get the appropriate storage backend based on project settings and file path
            storage_backend = get_storage_backend(project, file.file_path)

            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get the file from storage and save it to the temp directory
                file_obj = await storage_backend.get_file(file.file_path)
                temp_file_path = os.path.join(
                    temp_dir, os.path.basename(file.file_path)
                )

                # Save the file content to the temporary file
                with open(temp_file_path, "wb") as f:
                    f.write(file_obj.read())

                # Process the DICOM file
                results = process_dicom_batch(temp_dir)

                # The temporary directory and file will be automatically cleaned up

            # Update the file with the results
            file.processing_status = ProcessingStatus.COMPLETED
            file.processing_results = results
            file.processed_at = datetime.now(UTC)
            db.commit()

            logger.info(f"Successfully processed DICOM file: {file_id}")

        except DicomProcessingError as e:
            logger.error(f"Error processing DICOM file {file_id}: {str(e)}")
            file.processing_status = ProcessingStatus.FAILED
            file.processing_results = {"error": str(e)}
            file.processed_at = datetime.now(UTC)
            db.commit()

        except Exception as e:
            logger.error(f"Unexpected error processing DICOM file {file_id}: {str(e)}")
            logger.error(traceback.format_exc())
            file.processing_status = ProcessingStatus.FAILED
            file.processing_results = {"error": str(e)}
            file.processed_at = datetime.now(UTC)
            db.commit()

    except Exception as e:
        logger.error(f"Error in process_dicom_file: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(sys.argv) > 1:
        file_id = int(sys.argv[1])
        user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(process_dicom_file(file_id, user_id))
    else:
        print("Usage: python dicom_consumer.py <file_id> [<user_id>]")
