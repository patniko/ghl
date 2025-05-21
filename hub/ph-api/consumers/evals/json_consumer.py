import sys
import json
import asyncio
import traceback
from datetime import datetime, UTC
from loguru import logger
from sqlalchemy import select

from db import SessionLocal
from models import File as FileModel, ProcessingStatus, FileType, Project, ECGAnalysis
from services.storage import get_storage_backend


async def process_json_file(file_id: int, user_id: int):
    """
    Process a JSON file, specifically for AliveCor ECG data.

    Args:
        file_id: The ID of the JSON file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the File record
        file_stmt = select(FileModel).where(
            FileModel.id == file_id,
            FileModel.user_id == user_id,
            FileModel.file_type == FileType.JSON,
        )
        file_result = db.execute(file_stmt)
        file = file_result.scalar_one_or_none()

        if not file:
            logger.error(f"JSON file not found: file_id={file_id}, user_id={user_id}")
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

            # Get the appropriate storage backend
            storage = get_storage_backend(project, file.file_path)

            # Get the file from storage
            file_obj = await storage.get_file(file.file_path)

            # Read the file content
            content = file_obj.read()
            ecg_data = json.loads(content.decode("utf-8"))

            # Check if this is an AliveCor ECG file by looking for key indicators
            is_alivecor = False
            if isinstance(ecg_data, dict):
                # Check for typical AliveCor JSON structure
                if "data" in ecg_data and "raw" in ecg_data.get("data", {}):
                    is_alivecor = True
                # Additional checks could be added here

            if not is_alivecor:
                logger.warning(
                    f"File {file_id} does not appear to be an AliveCor ECG file"
                )
                file.processing_status = ProcessingStatus.COMPLETED
                file.processing_results = {
                    "message": "File processed but does not appear to be an AliveCor ECG file",
                    "is_alivecor": False,
                }
                file.schema_type = "unknown"
                file.processed_at = datetime.now(UTC)
                db.commit()
                return

            # Process the ECG data using ECGProcessor
            # processor = ECGProcessor()
            # analysis_results = processor.analyze_ecg(ecg_data)

            # # Get detailed neurokit2 analysis
            # detailed_analysis = processor.analyze_ecg_with_neurokit(ecg_data)

            # # Create ECGAnalysis record
            # ecg_analysis = ECGAnalysis(
            #     file_id=file.id,
            #     has_missing_leads=detailed_analysis["has_missing_leads"],
            #     signal_noise_ratio=detailed_analysis["signal_noise_ratio"],
            #     baseline_wander_score=detailed_analysis["baseline_wander_score"],
            #     motion_artifact_score=detailed_analysis["motion_artifact_score"],
            #     rr_interval_mean=detailed_analysis["rr_interval_mean"],
            #     rr_interval_stddev=detailed_analysis["rr_interval_stddev"],
            #     rr_interval_consistency=detailed_analysis["rr_interval_consistency"],
            #     qrs_count=detailed_analysis["qrs_count"],
            #     qrs_detection_confidence=detailed_analysis["qrs_detection_confidence"],
            #     hrv_sdnn=detailed_analysis["hrv_sdnn"],
            #     hrv_rmssd=detailed_analysis["hrv_rmssd"],
            #     hrv_pnn50=detailed_analysis["hrv_pnn50"],
            #     hrv_lf=detailed_analysis["hrv_lf"],
            #     hrv_hf=detailed_analysis["hrv_hf"],
            #     hrv_lf_hf_ratio=detailed_analysis["hrv_lf_hf_ratio"],
            #     frequency_peak=detailed_analysis["frequency_peak"],
            #     frequency_power_vlf=detailed_analysis["frequency_power_vlf"],
            #     frequency_power_lf=detailed_analysis["frequency_power_lf"],
            #     frequency_power_hf=detailed_analysis["frequency_power_hf"],
            # )
            # db.add(ecg_analysis)

            # Update file metadata with analysis results
            #file.file_metadata = {"ecg_analysis": analysis_results}
            # file.processing_status = ProcessingStatus.COMPLETED
            # file.processing_results = {
            #     "message": "AliveCor ECG file processed successfully",
            #     "is_alivecor": True,
            #     "analysis_summary": {
            #         "heart_rate": analysis_results.get("heart_rate"),
            #         "rhythm": analysis_results.get("analysis", {}).get("rhythm"),
            #         "abnormalities": analysis_results.get("analysis", {}).get(
            #             "abnormalities", []
            #         ),
            #         "signal_quality": {
            #             "signal_noise_ratio": detailed_analysis["signal_noise_ratio"],
            #             "baseline_wander": detailed_analysis["baseline_wander_score"],
            #             "motion_artifacts": detailed_analysis["motion_artifact_score"],
            #             "missing_leads": detailed_analysis["has_missing_leads"],
            #         },
            #     },
            # }
            # file.processed_at = datetime.now(UTC)
            # file.schema_type = "alivecor"
            # db.commit()

            logger.info(f"Successfully processed AliveCor ECG file: {file_id}")

        except Exception as e:
            logger.error(f"Error processing JSON file {file_id}: {str(e)}")
            logger.error(traceback.format_exc())
            file.processing_status = ProcessingStatus.FAILED
            file.processing_results = {"error": str(e)}
            file.processed_at = datetime.now(UTC)
            db.commit()

    except Exception as e:
        logger.error(f"Error in process_json_file: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(sys.argv) > 1:
        file_id = int(sys.argv[1])
        user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(process_json_file(file_id, user_id))
    else:
        print("Usage: python json_consumer.py <file_id> [<user_id>]")
