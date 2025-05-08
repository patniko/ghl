import sys
import asyncio
import traceback
from datetime import datetime, UTC
from loguru import logger
from sqlalchemy import select

from db import SessionLocal
from models import File, ProcessingStatus, FileType, Project
from services.storage import get_storage_backend


async def process_npz_file(file_id: int, user_id: int):
    """
    Process an NPZ (NumPy compressed) file.

    Args:
        file_id: The ID of the NPZ file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the NPZ file
        stmt = select(File).where(
            File.id == file_id, File.user_id == user_id, File.file_type == FileType.NPZ
        )

        result = db.execute(stmt)
        npz_file = result.scalar_one_or_none()

        if not npz_file:
            logger.error(f"NPZ file not found: file_id={file_id}, user_id={user_id}")
            return

        # Update status to processing
        npz_file.processing_status = ProcessingStatus.PROCESSING
        db.commit()

        try:
            # Get the project if file is associated with one
            project = None
            if npz_file.project_id:
                project_stmt = select(Project).where(Project.id == npz_file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()

            # Get the appropriate storage backend based on project settings and file path
            _ = get_storage_backend(project, npz_file.file_path)

            # Process the NPZ file
            logger.info(f"Processing NPZ file: {npz_file.file_path}")

            # In a real implementation, you would get the file from storage and process it
            # For example:
            # file_obj = await storage_backend.get_file(npz_file.file_path)
            # with io.BytesIO(file_obj.read()) as f:
            #     data = numpy.load(f)
            #     # Process the NumPy arrays

            # Stub implementation - in a real scenario, you would process the NPZ file here
            # For example, you might:
            # - Load the NumPy arrays from the file
            # - Perform data analysis or transformations
            # - Generate visualizations
            # - Extract metadata about the arrays

            # Simulate processing by waiting a short time
            await asyncio.sleep(1)

            # Update the file with the results
            npz_file.processing_status = ProcessingStatus.COMPLETED
            npz_file.processing_results = {
                "message": "NPZ processing completed successfully",
                "file_size_mb": round(npz_file.file_size / (1024 * 1024), 2),
                "metadata": {
                    "format": "npz",
                    "processed_at": datetime.now(UTC).isoformat(),
                    "arrays": [
                        "array1",
                        "array2",
                    ],  # Placeholder for actual array names
                },
            }
            npz_file.processed_at = datetime.now(UTC)
            db.commit()

            logger.info(f"Successfully processed NPZ file: {file_id}")

        except Exception as e:
            logger.error(f"Error processing NPZ file {file_id}: {str(e)}")
            logger.error(traceback.format_exc())
            npz_file.processing_status = ProcessingStatus.FAILED
            npz_file.processing_results = {"error": str(e)}
            npz_file.processed_at = datetime.now(UTC)
            db.commit()

    except Exception as e:
        logger.error(f"Error in process_npz_file: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(sys.argv) > 1:
        file_id = int(sys.argv[1])
        user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(process_npz_file(file_id, user_id))
    else:
        print("Usage: python npz_consumer.py <file_id> [<user_id>]")
