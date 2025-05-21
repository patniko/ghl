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
    Request,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from models import User, Batch, SampleDataset, Organization
from middleware.auth import get_organization_from_path

# Path to the samples.py script
SAMPLES_SCRIPT = Path(__file__).parent.parent.parent.parent / "tools" / "samples.py"

router = APIRouter()

async def generate_samples_dataset(
    dataset_id: int,
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
            SampleDataset.id == dataset_id
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
            
            # Store metadata in a temporary file
            metadata_file = output_dir / "metadata.json"
            metadata = {
                "data_types": data_types,
                "include_partials": include_partials,
                "partial_rate": partial_rate,
                "file_paths": file_paths,
                "generated_at": datetime.datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Update the dataset with a SQL update statement
            # Only update the num_patients field since that's all we have in the simplified model
            stmt = (
                update(SampleDataset)
                .where(SampleDataset.id == dataset_id)
                .values(
                    num_patients=num_patients
                )
            )
            db.execute(stmt)
            
            # Store the metadata in a separate file or database
            # For now, we'll just print it to the console
            print(f"Generated metadata for dataset {dataset_id}: {json.dumps(metadata)}")
            print(f"Generated data for dataset {dataset_id}: {len(data)} records")
            
            db.commit()
            
            print(f"Dataset {dataset_id} generated successfully")
    except Exception as e:
        print(f"Error generating dataset {dataset_id}: {e}")

class SampleDatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    num_patients: int
    batch_id: Optional[int] = None
    data_types: List[str] = ["questionnaire", "blood", "mobile", "consent", "echo", "ecg"]
    include_partials: bool = False
    partial_rate: float = 0.3
    output_dir: Optional[str] = None

class SampleDatasetResponse(BaseModel):
    id: int
    organization_id: int
    name: str
    description: Optional[str] = None
    num_patients: int
    created_at: datetime
    # These fields are not in the database model but are included in the response
    data_types: List[str] = []
    include_partials: bool = False
    partial_rate: float = 0.3
    output_dir: Optional[str] = None
    file_paths: Dict[str, List[str]] = {}

    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}

@router.post("/", response_model=SampleDatasetResponse)
async def create_samples_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    num_patients: int = Form(...),
    data_types: List[str] = Form(["questionnaire", "blood", "mobile", "consent", "echo", "ecg"]),
    include_partials: bool = Form(False),
    partial_rate: float = Form(0.3),
    batch_id: Optional[int] = Form(None),
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(get_organization_from_path),
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
    
    # Create initial metadata
    metadata = {
        "data_types": data_types,
        "include_partials": include_partials,
        "partial_rate": partial_rate,
        "file_paths": {},
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Create dataset record with only the fields that exist in the simplified model
    new_dataset = SampleDataset(
        organization_id=organization.id,
        name=name,
        description=description,
        num_patients=num_patients,
    )
    
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    
    # Start background task to generate data
    background_tasks.add_task(
        generate_samples_dataset,
        new_dataset.id,
        num_patients,
        data_types,
        include_partials,
        partial_rate,
        db,
    )
    
    # Create response with only the fields that exist in the simplified model
    # plus the additional fields we want to include in the response
    response_data = {
        "id": new_dataset.id,
        "organization_id": new_dataset.organization_id,
        "name": new_dataset.name,
        "description": new_dataset.description,
        "num_patients": new_dataset.num_patients,
        "created_at": new_dataset.created_at,
        "data_types": data_types,
        "include_partials": include_partials,
        "partial_rate": partial_rate,
        "output_dir": None,
        "file_paths": {},
    }
    
    return response_data


@router.get("/", response_model=List[SampleDatasetResponse])
async def get_samples_datasets(
    request: Request,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(get_organization_from_path),
    db: Session = Depends(get_db),
):
    """Get all synthetic datasets for the current user"""
    # Filter datasets by organization
    stmt = (
        select(SampleDataset)
        .where(SampleDataset.organization_id == organization.id)
        .order_by(SampleDataset.created_at.desc())
    )
    
    result = db.execute(stmt)
    datasets = result.scalars().all()
    
    # Convert SQLAlchemy models to response objects
    response_datasets = []
    for dataset in datasets:
        # Since we don't have these fields in the simplified model,
        # we'll use default values for the additional fields in the response
        response_data = {
            "id": dataset.id,
            "organization_id": dataset.organization_id,
            "name": dataset.name,
            "description": dataset.description,
            "num_patients": dataset.num_patients,
            "created_at": dataset.created_at,
            "data_types": [],  # Default empty list
            "include_partials": False,
            "partial_rate": 0.3,
            "output_dir": None,
            "file_paths": {},
        }
        response_datasets.append(response_data)
    
    return response_datasets


@router.get("/{dataset_id}", response_model=SampleDatasetResponse)
async def get_samples_dataset(
    dataset_id: int,
    request: Request,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(get_organization_from_path),
    db: Session = Depends(get_db),
):
    """Get a specific synthetic dataset"""
    # Get dataset by ID and organization
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.organization_id == organization.id
    )
    
    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create response with only the fields that exist in the simplified model
    # plus default values for the additional fields in the response
    response_data = {
        "id": dataset.id,
        "organization_id": dataset.organization_id,
        "name": dataset.name,
        "description": dataset.description,
        "num_patients": dataset.num_patients,
        "created_at": dataset.created_at,
        "data_types": [],  # Default empty list
        "include_partials": False,
        "partial_rate": 0.3,
        "output_dir": None,
        "file_paths": {},
    }
    
    return response_data


@router.delete("/{dataset_id}")
async def delete_samples_dataset(
    dataset_id: int,
    request: Request,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(get_organization_from_path),
    db: Session = Depends(get_db),
):
    """Delete a synthetic dataset"""
    # Get dataset by ID and organization
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.organization_id == organization.id
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
    request: Request,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(get_organization_from_path),
    db: Session = Depends(get_db),
):
    """Download a synthetic dataset as a zip file"""
    # Get dataset by ID and organization
    stmt = select(SampleDataset).where(
        SampleDataset.id == dataset_id,
        SampleDataset.organization_id == organization.id
    )
    
    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Since we don't have column_mappings in the simplified model,
    # we'll create a simple zip file with a JSON representation of the dataset
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Create a simple JSON file with the dataset information
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
                json_path = json_file.name
                dataset_info = {
                    "id": dataset.id,
                    "organization_id": dataset.organization_id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "num_patients": dataset.num_patients,
                    "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                }
                json.dump(dataset_info, json_file)
            
            zipf.write(json_path, "dataset_info.json")
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
