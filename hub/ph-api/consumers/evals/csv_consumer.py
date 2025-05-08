import sys
import asyncio
import traceback
import csv
import io
from datetime import datetime, UTC
from loguru import logger
from sqlalchemy import select

from db import SessionLocal
from models import File, ProcessingStatus, FileType, Check, DataType, Project
from services.catalog.checks.catalog import infer_column_data_type
from services.catalog.mappings.llm_mapping import analyze_dataset_columns_with_llm
from services.storage import get_storage_backend


async def infer_column_checks(file_obj, headers: list) -> dict:
    """
    Infer appropriate checks for each column in a CSV file based on the headers and sample data.

    Args:
        file_obj: File-like object containing CSV data
        headers: List of column headers

    Returns:
        Dictionary mapping column names to potential checks
    """
    # Read a sample of the CSV file to infer data types
    sample_data = []
    try:
        # Reset file pointer to beginning
        file_obj.seek(0)

        # Read as text
        text_content = io.TextIOWrapper(file_obj, encoding="utf-8")
        reader = csv.DictReader(text_content)

        # Get up to 100 rows for sampling
        for i, row in enumerate(reader):
            if i >= 100:  # Limit to 100 rows for performance
                break
            sample_data.append(row)

        # Reset file pointer for future use
        file_obj.seek(0)
    except Exception as e:
        logger.error(f"Error reading CSV file for inference: {str(e)}")
        return {}

    if not sample_data:
        logger.warning("No data found in CSV file for inference")
        return {}

    # Get all available checks from the database
    db = SessionLocal()
    try:
        checks_stmt = select(Check)
        checks_result = db.execute(checks_stmt)
        available_checks = checks_result.scalars().all()
    except Exception as e:
        logger.error(f"Error fetching checks: {str(e)}")
        available_checks = []
    finally:
        db.close()

    # Map columns to potential checks
    potential_mappings = {}

    for header in headers:
        # Extract values for this column from the sample data
        values = [row.get(header) for row in sample_data if row.get(header) is not None]

        # Infer data type
        data_type = infer_column_data_type(values)

        # Find applicable checks based on data type
        applicable_checks = [
            check for check in available_checks if check.data_type == data_type
        ]

        # Analyze the header name to suggest more specific checks
        header_lower = header.lower()
        suggested_checks = []

        # Add selected flag to each check
        for check in applicable_checks:
            check_info = {
                "id": check.id,
                "name": check.name,
                "description": check.description,
                "implementation": check.implementation,
                "parameters": check.parameters,
                "selected": False,  # Default to not selected
            }

            # Auto-select certain checks based on column name and data type
            if (
                any(term in header_lower for term in ["age", "year", "month", "day"])
                and data_type == DataType.NUMERIC
            ):
                if check.implementation == "check_range_compliance":
                    check_info["selected"] = True

            if (
                any(
                    term in header_lower
                    for term in ["date", "time", "birth", "admission", "discharge"]
                )
                and data_type == DataType.DATE
            ):
                if check.implementation == "check_date_range":
                    check_info["selected"] = True

            if (
                any(
                    term in header_lower
                    for term in ["name", "id", "identifier", "code"]
                )
                and data_type == DataType.TEXT
            ):
                if check.implementation == "check_missing_values":
                    check_info["selected"] = True

            if (
                any(
                    term in header_lower
                    for term in [
                        "gender",
                        "sex",
                        "race",
                        "ethnicity",
                        "status",
                        "type",
                        "category",
                    ]
                )
                and data_type == DataType.CATEGORICAL
            ):
                if check.implementation == "check_categorical_distribution":
                    check_info["selected"] = True

            if (
                any(
                    term in header_lower
                    for term in [
                        "weight",
                        "height",
                        "bmi",
                        "pressure",
                        "rate",
                        "count",
                        "level",
                        "score",
                    ]
                )
                and data_type == DataType.NUMERIC
            ):
                if check.implementation == "check_outliers":
                    check_info["selected"] = True

            # Always select missing values check for all columns
            if check.implementation == "check_missing_values":
                check_info["selected"] = True

            suggested_checks.append(check_info)

        # Store the mapping
        potential_mappings[header] = {
            "inferred_type": data_type,
            "applicable_checks": suggested_checks,
        }

    return potential_mappings


async def process_csv_file(file_id: int, user_id: int):
    """
    Process a CSV file.

    Args:
        file_id: The ID of the CSV file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the CSV file
        stmt = select(File).where(
            File.id == file_id, File.user_id == user_id, File.file_type == FileType.CSV
        )

        result = db.execute(stmt)
        csv_file = result.scalar_one_or_none()

        if not csv_file:
            logger.error(f"CSV file not found: file_id={file_id}, user_id={user_id}")
            return

        # Update status to processing
        csv_file.processing_status = ProcessingStatus.PROCESSING
        db.commit()

        try:
            # Process the CSV file
            logger.info(f"Processing CSV file: {csv_file.file_path}")

            # Infer checks based on headers
            if csv_file.csv_headers:
                logger.info(f"Inferring checks for CSV columns: {csv_file.csv_headers}")

                # Get all available checks from the database
                checks_stmt = select(Check)
                checks_result = db.execute(checks_stmt)
                available_checks = checks_result.scalars().all()

                # Get the project if file is associated with one
                project = None
                if csv_file.project_id:
                    project_stmt = select(Project).where(
                        Project.id == csv_file.project_id
                    )
                    project = db.execute(project_stmt).scalar_one_or_none()

                # Get the appropriate storage backend based on project settings
                storage_backend = get_storage_backend(project, csv_file.file_path)

                # Read a sample of the CSV file for LLM analysis
                sample_data = []
                try:
                    # Get the file from storage
                    file_obj = await storage_backend.get_file(csv_file.file_path)

                    # Read as text
                    text_content = io.TextIOWrapper(file_obj, encoding="utf-8")
                    reader = csv.DictReader(text_content)

                    # Get up to 5 rows for LLM analysis
                    for i, row in enumerate(reader):
                        if i >= 5:  # Limit to 5 rows for LLM
                            break
                        sample_data.append(row)

                    # Reset file pointer for future use
                    file_obj.seek(0)
                except Exception as e:
                    logger.error(f"Error reading CSV file for LLM analysis: {str(e)}")
                    sample_data = []

                if sample_data:
                    # Use LLM to analyze columns and suggest checks
                    logger.info("Using LLM to analyze columns and suggest checks")
                    try:
                        llm_analysis = await analyze_dataset_columns_with_llm(
                            headers=csv_file.csv_headers,
                            sample_rows=sample_data,
                            available_checks=available_checks,
                        )

                        # Convert LLM analysis to the format expected by the UI
                        potential_mappings = {}
                        column_mappings = llm_analysis.get("column_mappings", {})

                        for column_name, mapping in column_mappings.items():
                            inferred_type = mapping.get("inferred_data_type", "TEXT")
                            recommended_checks = mapping.get("recommended_checks", [])
                            reasoning = mapping.get("reasoning", "")

                            # Find the check objects for the recommended check IDs
                            applicable_checks = []
                            for check in available_checks:
                                check_info = {
                                    "id": check.id,
                                    "name": check.name,
                                    "description": check.description,
                                    "implementation": check.implementation,
                                    "parameters": check.parameters,
                                    "selected": check.id
                                    in recommended_checks,  # Auto-select recommended checks
                                }
                                applicable_checks.append(check_info)

                            potential_mappings[column_name] = {
                                "inferred_type": inferred_type,
                                "applicable_checks": applicable_checks,
                                "reasoning": reasoning,
                            }

                        logger.info(
                            f"LLM analysis completed with {len(potential_mappings)} column mappings"
                        )
                    except Exception as e:
                        logger.error(f"Error in LLM analysis: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Fall back to rule-based inference if LLM fails
                        # Get the file again since we need a fresh file object
                        file_obj = await storage_backend.get_file(csv_file.file_path)
                        potential_mappings = await infer_column_checks(
                            file_obj, csv_file.csv_headers
                        )
                else:
                    # Fall back to rule-based inference if no sample data
                    # Get the file again since we need a fresh file object
                    file_obj = await storage_backend.get_file(csv_file.file_path)
                    potential_mappings = await infer_column_checks(
                        file_obj, csv_file.csv_headers
                    )

                # Store the potential mappings
                csv_file.potential_mappings = potential_mappings
                logger.info(
                    f"Inferred potential mappings for {len(potential_mappings)} columns"
                )
            else:
                logger.warning(f"No CSV headers found for file {file_id}")

            # Update the file with the results
            csv_file.processing_status = ProcessingStatus.COMPLETED
            csv_file.processing_results = {
                "message": "CSV processing completed successfully",
                "rows_processed": len(csv_file.csv_headers)
                if csv_file.csv_headers
                else 0,
                "columns": csv_file.csv_headers,
                "potential_mappings_count": len(csv_file.potential_mappings)
                if csv_file.potential_mappings
                else 0,
            }
            csv_file.processed_at = datetime.now(UTC)
            db.commit()

            logger.info(f"Successfully processed CSV file: {file_id}")

        except Exception as e:
            logger.error(f"Error processing CSV file {file_id}: {str(e)}")
            logger.error(traceback.format_exc())
            csv_file.processing_status = ProcessingStatus.FAILED
            csv_file.processing_results = {"error": str(e)}
            csv_file.processed_at = datetime.now(UTC)
            db.commit()

    except Exception as e:
        logger.error(f"Error in process_csv_file: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(sys.argv) > 1:
        file_id = int(sys.argv[1])
        user_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(process_csv_file(file_id, user_id))
    else:
        print("Usage: python csv_consumer.py <file_id> [<user_id>]")
