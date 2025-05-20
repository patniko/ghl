import os
import sys
import random
import numpy as np
import pandas as pd
import datetime
import uuid
from pathlib import Path
import shutil
import zipfile
import tempfile
import json
import subprocess
from typing import List, Dict, Optional, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from models import User, Batch
from models_samples import SampleDataset, SampleDatasetCreate, SampleDatasetResponse

# Path to the samples.py script
SAMPLES_SCRIPT = Path(__file__).parent.parent.parent.parent / "tools" / "samples.py"

router = APIRouter()

async def generate_samples_dataset(
    dataset_id: int,
    user_id: int,
    num_patients: int,
    data_types: List[str],
    include_partials: bool,
    partial_rate: float,
    db: Session,
):
    """Background task to generate synthetic data using samples.py"""
    try:
        # Get the dataset
        stmt = select(SampleDataset).where(
            SampleDataset.id == dataset_id, SampleDataset.user_id == user_id
        )
        dataset = db.execute(stmt).scalar_one_or_none()
        
        if not dataset:
            print(f"Dataset {dataset_id} not found")
            return
        
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Build the command to run samples.py
            cmd = [
                sys.executable,
                str(SAMPLES_SCRIPT),
                "--num-samples", str(num_patients),
                "--output-dir", str(output_dir),
                "--type", ",".join(data_types),
            ]
            
            if include_partials:
                cmd.extend(["--include-partials", "--partial-rate", str(partial_rate)])
            
            # Create a zip file
            cmd.extend(["--create-zip"])
            
            # Run the command
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Get the output
            output = process.stdout
            
            # Find all generated files
            file_paths = {}
            for data_type in data_types:
                if data_type == "questionnaire":
                    csv_file = output_dir / "csv" / "questionnaire.csv"
                    if csv_file.exists():
                        if "csv" not in file_paths:
                            file_paths["csv"] = []
                        file_paths["csv"].append(str(csv_file))
                
                elif data_type == "blood":
                    csv_file = output_dir / "csv" / "blood_results.csv"
                    if csv_file.exists():
                        if "csv" not in file_paths:
                            file_paths["csv"] = []
                        file_paths["csv"].append(str(csv_file))
                
                elif data_type == "mobile":
                    csv_file = output_dir / "csv" / "mobile_measures.csv"
                    if csv_file.exists():
                        if "csv" not in file_paths:
                            file_paths["csv"] = []
                        file_paths["csv"].append(str(csv_file))
                
                elif data_type == "consent":
                    csv_file = output_dir / "csv" / "consent.csv"
                    if csv_file.exists():
                        if "csv" not in file_paths:
                            file_paths["csv"] = []
                        file_paths["csv"].append(str(csv_file))
                
                elif data_type == "echo":
                    echo_dir = output_dir / "echo"
                    if echo_dir.exists():
                        echo_files = list(echo_dir.glob("*.dcm"))
                        if echo_files:
                            file_paths["echo"] = [str(f) for f in echo_files]
                
                elif data_type == "ecg":
                    ecg_dir = output_dir / "ecg" / "normalized"
                    if ecg_dir.exists():
                        ecg_files = list(ecg_dir.glob("*.npy"))
                        if ecg_files:
                            file_paths["ecg"] = [str(f) for f in ecg_files]
            
            # Find the zip file
            zip_files = list(output_dir.glob("*.zip"))
            if zip_files:
                file_paths["zip"] = [str(zip_files[0])]
            
            # Read CSV files to get the data
            data = []
            if "csv" in file_paths:
                for csv_file in file_paths["csv"]:
                    try:
                        df = pd.read_csv(csv_file)
                        records = df.to_dict(orient="records")
                        data.extend(records)
                    except Exception as e:
                        print(f"Error reading CSV file {csv_file}: {e}")
            
            # Update the dataset
            dataset.data = data
            dataset.num_patients = num_patients
            dataset.column_mappings = {
                "data_types": data_types,
                "include_partials": include_partials,
                "partial_rate": partial_rate,
                "file_paths": file_paths,
            }
            
            db.commit()
            
            print(f"Dataset {dataset_id} generated successfully")
    except Exception as e:
        print(f"Error generating dataset {dataset_id}: {e}")

class SampleDatasetCreate(BaseModel):
    name: str
    description: str = None
    num_patients: int
    batch_id: int = None
    data_types: List[str] = ["questionnaire", "blood", "mobile", "consent", "echo", "ecg"]
    include_partials: bool = False
    partial_rate: float = 0.3
    output_dir: Optional[str] = None

class SampleDatasetResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: str = None
    num_patients: int
    batch_id: int = None
    column_mappings: dict = None
    applied_checks: dict = None
    check_results: dict = None
    created_at: datetime
    updated_at: datetime
    data_types: List[str]
    include_partials: bool
    partial_rate: float
    output_dir: Optional[str]
    file_paths: Dict[str, List[str]]

    model_config = {"from_attributes": True}

@router.post("/", response_model=SampleDatasetResponse)
async def create_samples_dataset(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(None),
    num_patients: int = Form(...),
    data_types: List[str] = Form(["questionnaire", "blood", "mobile", "consent", "echo", "ecg"]),
    include_partials: bool = Form(False),
    partial_rate: float = Form(0.3),
    batch_id: int = Form(None),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Create a new synthetic dataset using samples.py"""
    # Check if batch_id is provided and exists
    if batch_id:
        batch_stmt = select(Batch).where(
            Batch.id == batch_id, Batch.user_id == current_user["user_id"]
        )
        batch = db.execute(batch_stmt).scalar_one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
    
    # Validate data types
    valid_data_types = ["questionnaire", "blood", "mobile", "consent", "echo", "ecg", "all"]
    for data_type in data_types:
        if data_type not in valid_data_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data type: {data_type}. Valid types are: {', '.join(valid_data_types)}"
            )
    
    # Create dataset record
    new_dataset = SampleDataset(
        user_id=current_user["user_id"],
        batch_id=batch_id,
        name=name,
        description=description,
        num_patients=num_patients,
        data=[],  # Initially empty, will be populated by background task
        column_mappings={
            "data_types": data_types,
            "include_partials": include_partials,
            "partial_rate": partial_rate,
            "file_paths": {},
        },
        applied_checks={},
        check_results={},
    )
    
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    
    # Start background task to generate data
    background_tasks.add_task(
        generate_samples_dataset,
        new_dataset.id,
        current_user["user_id"],
        num_patients,
        data_types,
        include_partials,
        partial_rate,
        db,
    )
    
    # Create response
    response_data = {
        "id": new_dataset.id,
        "user_id": new_dataset.user_id,
        "batch_id": new_dataset.batch_id,
        "name": new_dataset.name,
        "description": new_dataset.description,
        "num_patients": new_dataset.num_patients,
        "created_at": new_dataset.created_at,
        "updated_at": new_dataset.updated_at,
        "column_mappings": new_dataset.column_mappings,
        "applied_checks": new_dataset.applied_checks,
        "check_results": new_dataset.check_results,
        "data_types": data_types,
        "include_partials": include_partials,
        "partial_rate": partial_rate,
        "output_dir": None,
        "file_paths": {},
    }
    
    return response_data


@router.get("/", response_model=List[SampleDatasetResponse])
async def get_samples_datasets(
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all synthetic datasets for the current user"""
    stmt = (
        select(SampleDataset)
        .where(SampleDataset.user_id == current_user["user_id"])
        .order_by(SampleDataset.created_at.desc())
    )
    
    result = db.execute(stmt)
    datasets = result.scalars().all()
    
    # Convert SQLAlchemy models to response objects
    response_datasets = []
    for dataset in datasets:
        # Extract data types and other info from column_mappings
        column_mappings = dataset.column_mappings or {}
        data_types = column_mappings.get("data_types", [])
        include_partials = column_mappings.get("include_partials", False)
        partial_rate = column_mappings.get("partial_rate", 0.3)
        output_dir = column_mappings.get("output_dir")
        file_paths = column_mappings.get("file_paths", {})
        
        response_data = {
            "id": dataset.id,
            "user_id": dataset.user_id,
            "batch_id": dataset.batch_id,
            "name": dataset.name,
            "description": dataset.description,
            "num_patients": dataset.num_patients,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "column_mappings": dataset.column_mappings,
            "applied_checks": dataset.applied_checks,
            "check_results": dataset.check_results,
            "data_types": data_types,
            "include_partials": include_partials,
            "partial_rate": partial_rate,
            "output_dir": output_dir,
            "file_paths": file_paths,
        }
        response_datasets.append(response_data)
    
    return response_datasets


@router.get("/{dataset_id}", response_model=SampleDatasetResponse)
async def get_samples_dataset(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific synthetic dataset"""
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.user_id == current_user["user_id"],
    )
    
    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Extract data types and other info from column_mappings
    column_mappings = dataset.column_mappings or {}
    data_types = column_mappings.get("data_types", [])
    include_partials = column_mappings.get("include_partials", False)
    partial_rate = column_mappings.get("partial_rate", 0.3)
    output_dir = column_mappings.get("output_dir")
    file_paths = column_mappings.get("file_paths", {})
    
    response_data = {
        "id": dataset.id,
        "user_id": dataset.user_id,
        "batch_id": dataset.batch_id,
        "name": dataset.name,
        "description": dataset.description,
        "num_patients": dataset.num_patients,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "column_mappings": dataset.column_mappings,
        "applied_checks": dataset.applied_checks,
        "check_results": dataset.check_results,
        "data_types": data_types,
        "include_partials": include_partials,
        "partial_rate": partial_rate,
        "output_dir": output_dir,
        "file_paths": file_paths,
    }
    
    return response_data


@router.delete("/{dataset_id}")
async def delete_samples_dataset(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a synthetic dataset"""
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.user_id == current_user["user_id"],
    )
    
    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    db.delete(dataset)
    db.commit()
    
    return {"message": "Dataset deleted successfully"}


@router.get("/{dataset_id}/download")
async def download_samples_dataset(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Download a synthetic dataset as a zip file"""
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.user_id == current_user["user_id"],
    )
    
    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Extract file paths from column_mappings
    column_mappings = dataset.column_mappings or {}
    file_paths = column_mappings.get("file_paths", {})
    
    # Check if there's a zip file
    if "zip" in file_paths and file_paths["zip"]:
        zip_path = file_paths["zip"][0]
        if os.path.exists(zip_path):
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{dataset.name.replace(' ', '_')}.zip",
            )
    
    # If no zip file, create one on the fly
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV files
            if "csv" in file_paths:
                for csv_path in file_paths["csv"]:
                    if os.path.exists(csv_path):
                        arcname = os.path.basename(csv_path)
                        zipf.write(csv_path, f"csv/{arcname}")
            
            # Add echo files
            if "echo" in file_paths:
                for echo_path in file_paths["echo"]:
                    if os.path.exists(echo_path):
                        arcname = os.path.basename(echo_path)
                        zipf.write(echo_path, f"echo/{arcname}")
            
            # Add ECG files
            if "ecg" in file_paths:
                for ecg_path in file_paths["ecg"]:
                    if os.path.exists(ecg_path):
                        arcname = os.path.basename(ecg_path)
                        zipf.write(ecg_path, f"ecg/normalized/{arcname}")
            
            # If no files were added, add the dataset data as JSON
            if not zipf.namelist() and dataset.data:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
                    json_path = json_file.name
                    json.dump(dataset.data, json_file)
                
                zipf.write(json_path, "data.json")
                os.unlink(json_path)
        
        return FileResponse(
            temp_path,
            media_type="application/zip",
            filename=f"{dataset.name.replace(' ', '_')}.zip",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating zip file: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
