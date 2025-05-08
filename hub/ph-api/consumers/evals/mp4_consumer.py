import sys
import asyncio
from datetime import datetime
import traceback
from loguru import logger
from sqlalchemy import select

from db import SessionLocal
from models import File, ProcessingStatus, FileType, Project
from services.storage import get_storage_backend


async def process_mp4_file(file_id: int, user_id: int):
    """
    Process an MP4 video file.

    Args:
        file_id: The ID of the MP4 file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the MP4 file
        stmt = select(File).where(
            File.id == file_id, File.user_id == user_id, File.file_type == FileType.MP4
        )

        result = db.execute(stmt)
        mp4_file = result.scalar_one_or_none()

        if not mp4_file:
            logger.error(f"MP4 file not found: file_id={file_id}, user_id={user_id}")
            return

        # Update status to processing
        mp4_file.processing_status = ProcessingStatus.PROCESSING
        db.commit()

        try:
            # Get the project if file is associated with one
            project = None
            if mp4_file.project_id:
                project_stmt = select(Project).where(Project.id == mp4_file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()

            # Get the appropriate storage backend based on project settings and file path
            _ = get_storage_backend(project, mp4_file.file_path)

            # Process the MP4 file
            logger.info(f"Processing MP4 file: {mp4_file.file_path}")

            # In a real implementation, you would get the file from storage and process it
            # For example:
            # file_obj = await storage_backend.get_file(mp4_file.file_path)
            # process_video(file_obj)

            # Stub implementation - in a real scenario, you would process the MP4 file here
            # For example, you might:
            # - Extract video metadata (duration, resolution, codec)
            # - Generate thumbnails
            # - Extract audio
            # - Perform video analysis
            # - Transcode to different formats

            # Simulate processing by waiting a short time
            await asyncio.sleep(1)

            # Update the file with the results
            mp4_file.processing_status = ProcessingStatus.COMPLETED
            mp4_file.processing_results = {
                "message": "MP4 processing completed successfully",
                "file_size_mb": round(mp4_file.file_size / (1024 * 1024), 2),
                "metadata": {
                    "format": "mp4",
                    "processed_at": datetime.utcnow().isoformat(),
                },
            }
            mp4_file.processed_at = datetime.utcnow()
            db.commit()

            logger.info(f"Successfully processed MP4 file: {file_id}")

        except Exception as e:
            logger.error(f"Error processing MP4 file {file_id}: {str(e)}")
            logger.error(traceback.format_exc())
            mp4_file.processing_status = ProcessingStatus.FAILED
            mp4_file.processing_results = {"error": str(e)}
            mp4_file.processed_at = datetime.utcnow()
            db.commit()

    except Exception as e:
        logger.error(f"Error in process_mp4_file: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(sys.argv) > 1:
        file_id = int(sys.argv[1])
        user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(process_mp4_file(file_id, user_id))
    else:
        print("Usage: python mp4_consumer.py <file_id> [<user_id>]")
