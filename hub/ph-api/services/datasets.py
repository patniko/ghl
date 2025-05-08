import random
import uuid
from typing import List, Dict

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
import asyncio
from auth import validate_jwt
from db import get_db
from models import (
    User,
    Batch,
    Organization,
    SyntheticDataset,
    SyntheticDatasetCreate,
    SyntheticDatasetResponse,
    SyntheticDatasetUpdate,
    Check,
)
from middleware import validate_user_organization


from fastapi import (
    File,
    UploadFile,
)

router = APIRouter()

# Store active WebSocket connections for progress updates
active_connections: Dict[int, List[WebSocket]] = {}


def generate_synthetic_patient():
    """
    Generate synthetic patient data mimicking the structure and distribution
    of the patientmetrics.csv dataset.

    Returns:
        dict: A dictionary containing synthetic patient data
    """
    # Initialize empty patient record
    patient = {}

    # Generate unique patient ID (UUID)
    patient["patient_ngsci_id"] = str(uuid.uuid4())

    # Year (current or recent)
    patient["year"] = random.choice([2022, 2023, 2024, 2025])

    # Consent is almost always "Yes"
    patient["verbal_consent"] = "Yes"

    # Age (follows distribution from data, mostly between 40-90)
    patient["age"] = random.randint(40, 90)

    # Sex (60% Female, 40% Male based on distribution)
    patient["sex"] = random.choices(["Female", "Male"], weights=[0.60, 0.40])[0]

    # Basic measurements
    # Most measurement fields have a "Yes" indicator and then the actual value

    # Blood pressure
    patient["bp"] = "Yes"
    patient["bp_systolic"] = round(
        random.normalvariate(127, 18), 0
    )  # Normal distribution around mean
    patient["bp_diastolic"] = round(random.normalvariate(79, 11), 0)

    # Pulse
    patient["pulse"] = "Yes"
    patient["pulse_entry"] = round(random.normalvariate(80, 10), 0)

    # Respiratory rate
    patient["resp_rate"] = "Yes"
    patient["resp_rate_entry"] = round(random.normalvariate(20, 4), 0)

    # Oxygen saturation
    patient["spo2"] = "Yes"
    patient["spo2_entry"] = min(100, round(random.normalvariate(97, 2), 0))

    # Random blood sugar
    patient["rbs"] = "Yes"
    patient["rbs_entry"] = round(random.normalvariate(120, 40), 1)

    # Height (cm)
    patient["height"] = "Yes"
    # Males tend to be taller
    if patient["sex"] == "Male":
        patient["height_entry"] = round(random.normalvariate(170, 8), 0)
    else:
        patient["height_entry"] = round(random.normalvariate(155, 7), 0)

    # Weight (kg)
    patient["weight"] = "Yes"
    # Weight correlates with height
    bmi_factor = random.normalvariate(24, 5)  # Average BMI with variation
    height_m = patient["height_entry"] / 100
    patient["weight_entry"] = round(bmi_factor * (height_m * height_m), 1)

    # Mid-arm circumference
    patient["midarm_circum"] = "Yes"
    patient["midarm_circum_entry"] = round(random.normalvariate(28, 4), 0)

    # Waist circumference
    patient["waist_circum"] = "Yes"
    if patient["sex"] == "Male":
        patient["waist_circum_entry"] = round(random.normalvariate(90, 12), 0)
    else:
        patient["waist_circum_entry"] = round(random.normalvariate(85, 12), 0)

    # Hip circumference
    patient["hip_circum"] = "Yes"
    patient["hip_circum_entry"] = round(
        patient["waist_circum_entry"] * random.uniform(1.05, 1.25), 0
    )

    # Endurance test
    patient["endurance_test"] = "Yes"
    patient["endurance_test_entry"] = max(0, round(random.normalvariate(12, 5), 0))

    # Grip strength
    patient["grip_left"] = "Yes"
    if patient["sex"] == "Male":
        patient["grip_left_entry"] = round(random.normalvariate(30, 8), 1)
    else:
        patient["grip_left_entry"] = round(random.normalvariate(20, 6), 1)

    patient["grip_right"] = "Yes"
    patient["grip_right_entry"] = round(
        patient["grip_left_entry"] * random.uniform(1.0, 1.2), 1
    )  # Right grip usually stronger

    # Eye measurements
    patient["tonometry_lefteye"] = "Yes"
    patient["tonometry_lefteye_entry"] = round(random.normalvariate(14, 3), 1)

    patient["tonometry_righteye"] = "Yes"
    patient["tonometry_righteye_entry"] = round(
        random.normalvariate(patient["tonometry_lefteye_entry"], 1), 1
    )

    patient["fundus_lefteye"] = "Yes"
    patient["fundus_lefteye_obs"] = None  # Often null in the data

    patient["fundus_righteye"] = "Yes"
    patient["fundus_righteye_obs"] = None  # Often null in the data

    # Cognitive assessment
    patient["cognition_sf"] = "Yes"
    patient["cognition_sf_score"] = round(random.normalvariate(16, 4), 0)

    # Cognitive impairment (binary outcome based on cognition score)
    patient["cognit_impaired"] = 1 if patient["cognition_sf_score"] < 10 else 0

    # Lab values
    # Hemoglobin
    if patient["sex"] == "Male":
        patient["Hb"] = round(random.normalvariate(14, 1.5), 1)
    else:
        patient["Hb"] = round(random.normalvariate(12, 1.5), 1)

    # HbA1c (glycated hemoglobin)
    patient["HbA1c"] = round(random.normalvariate(6, 1.5), 1)

    # Lipid profile
    patient["triglycerides_mg_dl"] = round(random.normalvariate(140, 70), 0)
    patient["tot_cholesterol_mg_dl"] = round(random.normalvariate(190, 40), 0)
    patient["HDL_mg_dl"] = round(random.normalvariate(50, 15), 0)
    patient["LDL_mg_dl"] = round(random.normalvariate(110, 30), 0)
    patient["VLDL_mg_dl"] = round(
        patient["triglycerides_mg_dl"] / 5, 1
    )  # VLDL is often calculated as TG/5

    # Cholesterol ratios
    patient["totchol_by_hdl_ratio"] = round(
        patient["tot_cholesterol_mg_dl"] / patient["HDL_mg_dl"], 2
    )
    patient["ldl_by_hdl_ratio"] = round(patient["LDL_mg_dl"] / patient["HDL_mg_dl"], 2)

    # Creatinine
    if patient["sex"] == "Male":
        patient["creatinine_mg_dl"] = round(random.normalvariate(1.0, 0.2), 1)
    else:
        patient["creatinine_mg_dl"] = round(random.normalvariate(0.8, 0.2), 1)

    # Education/literacy status
    patient["literate"] = random.choices(["Yes", "No"], weights=[0.62, 0.38])[0]

    # Smoking status (based on distribution from data)
    patient["smoking_1"] = random.choices(
        ["Not at all", "Daily", "Less than daily", "Don't know", "Refused"],
        weights=[0.85, 0.058, 0.036, 0.037, 0.019],
    )[0]

    # Smoking follow-up questions
    if patient["smoking_1"] in ["Not at all", "Refused", "Don't know"]:
        patient["smoking_2"] = None
    else:
        patient["smoking_2"] = random.choice(["Yes", "No", "Don't know"])

    patient["smoking_3"] = random.choices(
        ["Not at all", "Daily", "Less than daily", "Don't know", "Refused"],
        weights=[0.85, 0.058, 0.036, 0.037, 0.019],
    )[0]

    # PHQ (Patient Health Questionnaire) responses
    phq_options = [
        "Not at all",
        "Several days",
        "More than half the days",
        "Nearly everyday",
    ]
    phq_weights = [0.28, 0.28, 0.23, 0.19]

    for i in range(1, 5):
        patient[f"phq_{i}"] = random.choices(phq_options, weights=phq_weights)[0]

    # Loneliness assessment
    patient["direct_lonely"] = random.choices(phq_options, weights=phq_weights)[0]

    return patient


async def generate_dataset_task(
    dataset_id: int,
    user_id: int,
    organization_id: int,
    num_patients: int,
    db: Session,
):
    """Background task to generate synthetic patient data"""
    # Generate patients in batches to avoid memory issues
    batch_size = min(100, num_patients)  # Process in batches of 100 or less
    all_patients = []

    for i in range(0, num_patients, batch_size):
        # Calculate the actual batch size (might be smaller for the last batch)
        current_batch_size = min(batch_size, num_patients - i)

        # Generate batch of patients
        batch_patients = [
            generate_synthetic_patient() for _ in range(current_batch_size)
        ]
        all_patients.extend(batch_patients)

        # Update progress via WebSocket
        progress = min(100, int((i + current_batch_size) / num_patients * 100))
        if user_id in active_connections:
            for connection in active_connections[user_id]:
                try:
                    await connection.send_json(
                        {
                            "dataset_id": dataset_id,
                            "progress": progress,
                            "completed": progress == 100,
                            "current": i + current_batch_size,
                            "total": num_patients,
                        }
                    )
                except Exception:
                    # Connection might be closed
                    pass

        # Small delay to prevent CPU overload and allow for cancellation
        await asyncio.sleep(0.01)

    # Update the dataset with the generated data
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == user_id,
        SyntheticDataset.organization_id == organization_id,
    )
    dataset = db.execute(stmt).scalar_one_or_none()

    if dataset:
        dataset.data = all_patients
        db.commit()

    # Final progress update
    if user_id in active_connections:
        for connection in active_connections[user_id]:
            try:
                await connection.send_json(
                    {
                        "dataset_id": dataset_id,
                        "progress": 100,
                        "completed": True,
                        "current": num_patients,
                        "total": num_patients,
                    }
                )
            except Exception:
                # Connection might be closed
                pass


@router.post("/", response_model=SyntheticDatasetResponse)
async def create_dataset(
    dataset: SyntheticDatasetCreate,
    background_tasks: BackgroundTasks,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Create a new synthetic dataset"""
    # Check if batch_id is provided and exists
    batch_id = dataset.batch_id
    if batch_id:
        batch_stmt = select(Batch).where(
            Batch.id == batch_id, Batch.user_id == current_user["user_id"]
        )
        batch = db.execute(batch_stmt).scalar_one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
    else:
        # Set batch_id to 0 instead of None to avoid validation errors
        batch_id = 0

    # Create dataset record
    new_dataset = SyntheticDataset(
        user_id=current_user["user_id"],
        organization_id=organization.id,  # Set organization_id from the validated organization
        batch_id=batch_id if batch_id != 0 else None,  # Store as None in DB if it was 0
        name=dataset.name,
        description=dataset.description,
        num_patients=dataset.num_patients,
        data=[],  # Initially empty, will be populated by background task
        column_mappings={},  # Will be populated when data is analyzed
        applied_checks={},  # Will be populated when checks are applied
        check_results={},  # Will be populated when checks are run
    )

    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    # Start background task to generate data
    background_tasks.add_task(
        generate_dataset_task,
        new_dataset.id,
        current_user["user_id"],
        organization.id,
        dataset.num_patients,
        db,
    )

    # Create a response object with batch_id as 0 instead of None
    response_data = {
        "id": new_dataset.id,
        "organization_id": new_dataset.organization_id,
        "user_id": new_dataset.user_id,
        "batch_id": 0 if new_dataset.batch_id is None else new_dataset.batch_id,
        "name": new_dataset.name,
        "description": new_dataset.description,
        "num_patients": new_dataset.num_patients,
        "created_at": new_dataset.created_at,
        "updated_at": new_dataset.updated_at,
        "column_mappings": new_dataset.column_mappings,
        "applied_checks": new_dataset.applied_checks,
        "check_results": new_dataset.check_results,
    }
    return response_data


@router.get("/", response_model=List[SyntheticDatasetResponse])
async def get_datasets(
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all synthetic datasets for the current user"""
    stmt = (
        select(SyntheticDataset)
        .where(
            SyntheticDataset.user_id == current_user["user_id"],
            SyntheticDataset.organization_id == organization.id,
        )
        .order_by(SyntheticDataset.created_at.desc())
    )

    result = db.execute(stmt)
    datasets = result.scalars().all()

    # Convert SQLAlchemy models to dictionaries and ensure batch_id is 0 instead of None
    response_datasets = []
    for dataset in datasets:
        response_data = {
            "id": dataset.id,
            "organization_id": dataset.organization_id,
            "user_id": dataset.user_id,
            "batch_id": 0 if dataset.batch_id is None else dataset.batch_id,
            "name": dataset.name,
            "description": dataset.description,
            "num_patients": dataset.num_patients,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "column_mappings": dataset.column_mappings,
            "applied_checks": dataset.applied_checks,
            "check_results": dataset.check_results,
        }
        response_datasets.append(response_data)

    return response_datasets


@router.get("/{dataset_id}", response_model=SyntheticDatasetResponse)
async def get_dataset(
    dataset_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific synthetic dataset"""
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create a response object with batch_id as 0 instead of None
    response_data = {
        "id": dataset.id,
        "organization_id": dataset.organization_id,
        "user_id": dataset.user_id,
        "batch_id": 0 if dataset.batch_id is None else dataset.batch_id,
        "name": dataset.name,
        "description": dataset.description,
        "num_patients": dataset.num_patients,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "column_mappings": dataset.column_mappings,
        "applied_checks": dataset.applied_checks,
        "check_results": dataset.check_results,
    }
    return response_data


@router.get("/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: int,
    page: int = 1,
    page_size: int = 100,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get paginated data from a specific synthetic dataset"""
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Calculate pagination
    data = dataset.data or []
    total = len(data)
    total_pages = (total + page_size - 1) // page_size

    # Validate page number
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages

    # Get paginated data
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    paginated_data = data[start_idx:end_idx] if start_idx < total else []

    return {
        "data": paginated_data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": total_pages,
        },
    }


@router.put("/{dataset_id}", response_model=SyntheticDatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: SyntheticDatasetUpdate,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Update a synthetic dataset's metadata"""
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Update fields
    if dataset_update.name is not None:
        dataset.name = dataset_update.name
    if dataset_update.description is not None:
        dataset.description = dataset_update.description
    if dataset_update.batch_id is not None:
        # Check if batch exists
        if dataset_update.batch_id > 0:
            batch_stmt = select(Batch).where(
                Batch.id == dataset_update.batch_id,
                Batch.user_id == current_user["user_id"],
            )
            batch = db.execute(batch_stmt).scalar_one_or_none()
            if not batch:
                raise HTTPException(status_code=404, detail="Batch not found")
            dataset.batch_id = dataset_update.batch_id
        else:
            dataset.batch_id = None
    if dataset_update.column_mappings is not None:
        dataset.column_mappings = dataset_update.column_mappings
    if dataset_update.applied_checks is not None:
        dataset.applied_checks = dataset_update.applied_checks

    db.commit()
    db.refresh(dataset)

    # Create a response object with batch_id as 0 instead of None
    response_data = {
        "id": dataset.id,
        "organization_id": dataset.organization_id,
        "user_id": dataset.user_id,
        "batch_id": 0 if dataset.batch_id is None else dataset.batch_id,
        "name": dataset.name,
        "description": dataset.description,
        "num_patients": dataset.num_patients,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "column_mappings": dataset.column_mappings,
        "applied_checks": dataset.applied_checks,
        "check_results": dataset.check_results,
    }
    return response_data


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a synthetic dataset"""
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()

    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/upload-csv")
async def upload_csv(
    dataset_id: int,
    file: UploadFile = File(...),
    analyze_with_llm: bool = True,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file to replace the synthetic dataset data.
    Optionally analyze the dataset with LLM to suggest column-to-check mappings.
    """
    import pandas as pd
    import io
    from services.catalog.mappings.llm_mapping import analyze_uploaded_dataset

    # Get the dataset
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Read file content
        file_content = await file.read()

        # Parse CSV
        df = pd.read_csv(io.BytesIO(file_content))

        # Convert to list of dictionaries
        records = df.to_dict(orient="records")

        # Update dataset
        dataset.data = records
        dataset.num_patients = len(records)

        # Reset column mappings and check results
        dataset.column_mappings = {}
        dataset.applied_checks = {}
        dataset.check_results = {}

        db.commit()

        response_data = {
            "message": "CSV uploaded successfully",
            "num_records": len(records),
            "columns": list(df.columns),
        }

        # If LLM analysis is requested, perform it
        if analyze_with_llm and records:
            try:
                # Get all available checks
                checks_stmt = select(Check)
                checks_result = db.execute(checks_stmt)
                available_checks = checks_result.scalars().all()

                # Analyze the dataset with LLM
                analysis_result = await analyze_uploaded_dataset(
                    data=records, available_checks=available_checks
                )

                # Update dataset with column mappings
                dataset.column_mappings = analysis_result.get("column_mappings", {})

                # Apply the suggested checks
                check_mappings = analysis_result.get("check_mappings", {})
                if check_mappings:
                    dataset.applied_checks = {
                        column_name: [
                            {
                                "id": check_id,
                                "name": next(
                                    (
                                        check.name
                                        for check in available_checks
                                        if check.id == check_id
                                    ),
                                    "Unknown Check",
                                ),
                                "parameters": next(
                                    (
                                        check.parameters
                                        for check in available_checks
                                        if check.id == check_id
                                    ),
                                    {},
                                ),
                            }
                            for check_id in check_ids
                        ]
                        for column_name, check_ids in check_mappings.items()
                    }

                db.commit()

                # Add analysis results to response
                response_data["analysis"] = {
                    "column_mappings": analysis_result.get("column_mappings", {}),
                    "applied_checks": dataset.applied_checks,
                }
            except Exception as e:
                # Log the error but don't fail the upload
                print(f"Error during LLM analysis: {str(e)}")
                response_data["analysis_error"] = str(e)

        return response_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.post("/{dataset_id}/infer-column-types")
async def infer_column_types(
    dataset_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Infer column types for a dataset"""
    # Get the dataset
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
        SyntheticDataset.organization_id == organization.id,
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get the data
    data = dataset.data or []

    if not data:
        return {"message": "No data available for analysis", "column_types": {}}

    # Import the function from catalog
    from services.catalog.checks.catalog import infer_column_data_type

    # Infer column types
    column_types = {}
    for column_name in data[0].keys():
        # Extract values for this column
        values = [patient.get(column_name) for patient in data]

        # Infer data type
        data_type = infer_column_data_type(values)
        column_types[column_name] = data_type

    # Update dataset column mappings
    dataset.column_mappings = {
        column_name: {
            "inferred_type": data_type,
            "mapped_type": data_type,  # Default to inferred type
            "description": None,
        }
        for column_name, data_type in column_types.items()
    }

    db.commit()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "column_types": column_types,
    }


@router.websocket("/ws/progress")
async def websocket_progress(
    websocket: WebSocket,
    user_id: int,
):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()

    # Add connection to active connections
    if user_id not in active_connections:
        active_connections[user_id] = []
    active_connections[user_id].append(websocket)

    try:
        # Keep connection open until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Remove connection when client disconnects
        if user_id in active_connections:
            active_connections[user_id].remove(websocket)
            if not active_connections[user_id]:
                del active_connections[user_id]
