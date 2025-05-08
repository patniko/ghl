import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from middleware import validate_user_organization
from models import (
    User,
    Batch,
    Organization,
    BatchCreate,
    BatchResponse,
    BatchUpdate,
    SyntheticDataset,
    DicomFile,
    File,
    FileResponse,
    Project,
)
from services.storage import get_storage_backend

router = APIRouter()


@router.get("/statistics")
async def get_batch_statistics(
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get aggregated statistics for all batches in the organization"""
    # Get total number of batches
    batches_stmt = (
        select(func.count())
        .select_from(Batch)
        .where(
            Batch.organization_id == organization.id,
            Batch.user_id == current_user["user_id"],
        )
    )
    total_batches = db.execute(batches_stmt).scalar() or 0

    # Get total number of datasets
    datasets_stmt = (
        select(func.count())
        .select_from(SyntheticDataset)
        .where(
            SyntheticDataset.organization_id == organization.id,
            SyntheticDataset.user_id == current_user["user_id"],
        )
    )
    total_datasets = db.execute(datasets_stmt).scalar() or 0

    # Get total number of DICOM batches
    total_dicom_batches = 0

    # Get total number of DICOM files
    dicom_files_stmt = (
        select(func.count())
        .select_from(DicomFile)
        .where(
            DicomFile.organization_id == organization.id,
            DicomFile.user_id == current_user["user_id"],
        )
    )
    total_dicom_files = db.execute(dicom_files_stmt).scalar() or 0

    # Calculate data quality
    # For this example, we'll use a simple calculation based on dataset check results
    # In a real app, this would be more sophisticated
    data_quality = 85  # Default value

    # Get the most recent update timestamp
    last_updated_stmt = select(func.max(Batch.updated_at)).where(
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )
    last_updated = db.execute(last_updated_stmt).scalar()

    return {
        "totalBatches": total_batches,
        "totalDatasets": total_datasets,
        "totalDicomBatches": total_dicom_batches,
        "totalDicomFiles": total_dicom_files,
        "dataQuality": data_quality,
        "lastUpdated": last_updated.isoformat() if last_updated else None,
    }


@router.post("", response_model=BatchResponse)
async def create_batch(
    batch: BatchCreate,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Create a new batch"""
    # Check if project exists if project_id is provided
    project = None
    if batch.project_id:
        project_stmt = select(Project).where(
            Project.id == batch.project_id,
            Project.organization_id == organization.id,
            Project.user_id == current_user["user_id"],
        )
        project = db.execute(project_stmt).scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    new_batch = Batch(
        organization_id=organization.id,
        user_id=current_user["user_id"],
        project_id=batch.project_id,
        name=batch.name,
        description=batch.description,
        quality_summary={},  # Initialize with empty dictionary to avoid validation errors
    )

    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    # Create batch folder if project is provided
    if project:
        try:
            storage_backend = get_storage_backend(project)
            batch_path = f"projects/{project.id}/batches/{new_batch.id}"
            
            # Handle different storage backends
            if hasattr(storage_backend, 'base_dir'):
                # Local storage backend
                os.makedirs(os.path.join(storage_backend.base_dir, batch_path), exist_ok=True)
            else:
                # S3 storage backend - create empty object to represent folder
                try:
                    # For S3, we create an empty object with trailing slash to represent folder
                    storage_backend.s3_client.put_object(
                        Bucket=storage_backend.bucket_name,
                        Key=f"{batch_path}/"
                    )
                except Exception as s3_error:
                    print(f"Error creating S3 folder: {str(s3_error)}")
        except Exception as e:
            # Log the error but don't fail the batch creation
            print(f"Error creating batch folder: {str(e)}")

    return new_batch


@router.get("", response_model=List[BatchResponse])
async def get_batches(
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all batches for the current user in the organization"""
    stmt = (
        select(Batch)
        .where(
            Batch.organization_id == organization.id,
            Batch.user_id == current_user["user_id"],
        )
        .order_by(Batch.created_at.desc())
    )

    result = db.execute(stmt)
    batches = result.scalars().all()

    # Ensure all batches have a valid quality_summary
    for batch in batches:
        if batch.quality_summary is None:
            batch.quality_summary = {
                "total_datasets": 0,
                "total_checks": 0,
                "checks_by_type": {},
                "issues_by_severity": {"info": 0, "warning": 0, "error": 0},
            }
            db.add(batch)

    # Commit any changes
    if any(batch.quality_summary is None for batch in batches):
        db.commit()

    return batches


@router.get("/all-files")
async def get_all_batch_files(
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    file_type: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc",
    page: Optional[int] = 1,
    page_size: Optional[int] = 20,
):
    """Get all files across all batches with pagination, search, and sorting"""
    # Base query for files
    query = select(File).where(
        File.organization_id == organization.id,
        File.user_id == current_user["user_id"],
    )

    # Apply project filter if provided
    if project_id:
        # First check if the project exists and belongs to the user and organization
        project_stmt = select(Project).where(
            Project.id == project_id,
            Project.organization_id == organization.id,
            Project.user_id == current_user["user_id"],
        )
        project_result = db.execute(project_stmt)
        project = project_result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get all batches for this project
        batch_stmt = select(Batch.id).where(
            Batch.project_id == project_id,
            Batch.organization_id == organization.id,
        )
        batch_result = db.execute(batch_stmt)
        batch_ids = [batch[0] for batch in batch_result]

        if batch_ids:
            query = query.where(File.batch_id.in_(batch_ids))
        else:
            # If no batches found, return empty list
            return {
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "files": [],
            }

    # Apply file type filter if provided
    if file_type:
        query = query.where(File.file_type == file_type)

    # Apply search filter if provided
    if search:
        search_term = f"%{search}%"
        query = query.where(File.original_filename.ilike(search_term))

    # Apply sorting
    if sort_by == "filename":
        sort_column = File.original_filename
    elif sort_by == "file_size":
        sort_column = File.file_size
    elif sort_by == "file_type":
        sort_column = File.file_type
    else:  # Default to created_at
        sort_column = File.created_at

    if sort_order.lower() == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Count total files for pagination
    count_query = select(func.count()).select_from(query.subquery())
    total_files = db.execute(count_query).scalar() or 0

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Execute query
    files_result = db.execute(query)
    files = files_result.scalars().all()

    # Calculate total pages
    total_pages = (total_files + page_size - 1) // page_size

    # Get batch information for each file
    file_responses = []
    for file in files:
        # Get batch information
        batch_stmt = select(Batch).where(Batch.id == file.batch_id)
        batch_result = db.execute(batch_stmt)
        batch = batch_result.scalar_one_or_none()

        file_dict = {
            **file.__dict__,
            "batch_name": batch.name if batch else None,
            "project_name": None,
        }

        # Get project information if available
        if batch and batch.project_id:
            project_stmt = select(Project).where(Project.id == batch.project_id)
            project_result = db.execute(project_stmt)
            project = project_result.scalar_one_or_none()
            if project:
                file_dict["project_name"] = project.name

        # Remove SQLAlchemy state attributes
        file_dict.pop("_sa_instance_state", None)

        file_responses.append(file_dict)

    return {
        "total": total_files,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "files": file_responses,
    }


@router.get("/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific batch"""
    stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    result = db.execute(stmt)
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Ensure batch has a valid quality_summary
    if batch.quality_summary is None:
        batch.quality_summary = {
            "total_datasets": 0,
            "total_checks": 0,
            "checks_by_type": {},
            "issues_by_severity": {"info": 0, "warning": 0, "error": 0},
        }
        db.add(batch)
        db.commit()

    return batch


@router.put("/{batch_id}", response_model=BatchResponse)
async def update_batch(
    batch_id: int,
    batch_update: BatchUpdate,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Update a batch"""
    stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    result = db.execute(stmt)
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Update fields
    if batch_update.name is not None:
        batch.name = batch_update.name
    if batch_update.description is not None:
        batch.description = batch_update.description

    db.commit()
    db.refresh(batch)

    return batch


@router.delete("/{batch_id}")
async def delete_batch(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a batch and clean up its folder"""
    stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    result = db.execute(stmt)
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all files associated with this batch
    files_stmt = select(File).where(
        File.batch_id == batch_id,
        File.organization_id == organization.id,
    )
    files_result = db.execute(files_stmt)
    files = files_result.scalars().all()

    # Get the project if batch is associated with one
    project = None
    if batch.project_id:
        project_stmt = select(Project).where(Project.id == batch.project_id)
        project = db.execute(project_stmt).scalar_one_or_none()

    # Get the appropriate storage backend based on project settings
    storage_backend = get_storage_backend(project)

    # Delete all files from storage
    for file in files:
        try:
            await storage_backend.delete_file(file.file_path)
        except Exception as e:
            # Log the error but continue with deletion
            print(f"Error deleting file {file.file_path}: {str(e)}")

    # Delete the batch folder if it exists
    if project:
        try:
            batch_path = f"projects/{project.id}/batches/{batch.id}"
            
            # Handle different storage backends
            if hasattr(storage_backend, 'base_dir'):
                # Local storage backend
                batch_dir = os.path.join(storage_backend.base_dir, batch_path)
                if os.path.exists(batch_dir):
                    import shutil
                    shutil.rmtree(batch_dir)
            else:
                # S3 storage backend - delete objects with the batch path prefix
                try:
                    # List all objects with the batch path prefix
                    objects = storage_backend.s3_client.list_objects_v2(
                        Bucket=storage_backend.bucket_name,
                        Prefix=batch_path
                    )
                    
                    # Delete all objects with the batch path prefix
                    if 'Contents' in objects:
                        for obj in objects['Contents']:
                            storage_backend.s3_client.delete_object(
                                Bucket=storage_backend.bucket_name,
                                Key=obj['Key']
                            )
                except Exception as s3_error:
                    print(f"Error deleting S3 folder: {str(s3_error)}")
        except Exception as e:
            # Log the error but continue with deletion
            print(f"Error deleting batch folder: {str(e)}")

    # Delete all files from database
    for file in files:
        db.delete(file)

    # Delete the batch from database
    db.delete(batch)
    db.commit()

    return {"message": "Batch deleted successfully"}


@router.get("/{batch_id}/datasets")
async def get_batch_datasets(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all datasets associated with a batch"""
    # First check if the batch exists and belongs to the user and organization
    batch_stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    batch_result = db.execute(batch_stmt)
    batch = batch_result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all datasets for this batch
    datasets_stmt = (
        select(SyntheticDataset)
        .where(
            SyntheticDataset.batch_id == batch_id,
            SyntheticDataset.organization_id == organization.id,
            SyntheticDataset.user_id == current_user["user_id"],
        )
        .order_by(SyntheticDataset.created_at.desc())
    )

    datasets_result = db.execute(datasets_stmt)
    datasets = datasets_result.scalars().all()

    return datasets


@router.get("/{batch_id}/files", response_model=List[FileResponse])
async def get_batch_files(
    batch_id: int,
    file_type: Optional[str] = None,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all files associated with a batch"""
    # First check if the batch exists and belongs to the user and organization
    batch_stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    batch_result = db.execute(batch_stmt)
    batch = batch_result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all files for this batch
    files_stmt = select(File).where(
        File.batch_id == batch_id,
        File.organization_id == organization.id,
        File.user_id == current_user["user_id"],
    )

    # Apply file type filter if provided
    if file_type:
        files_stmt = files_stmt.where(File.file_type == file_type)

    # Order by creation date, newest first
    files_stmt = files_stmt.order_by(File.created_at.desc())

    files_result = db.execute(files_stmt)
    files = files_result.scalars().all()

    return files


@router.post("/{project_id}/create-new-batch", response_model=BatchResponse)
async def create_new_batch_for_project(
    project_id: int,
    batch_name: Optional[str] = None,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Create a new batch for a project and cycle to it"""
    # Check if project exists
    project_stmt = select(Project).where(
        Project.id == project_id,
        Project.organization_id == organization.id,
        Project.user_id == current_user["user_id"],
    )
    project = db.execute(project_stmt).scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get the count of existing batches for this project to generate a name
    batch_count_stmt = (
        select(func.count())
        .select_from(Batch)
        .where(
            Batch.project_id == project_id,
            Batch.organization_id == organization.id,
        )
    )
    batch_count = db.execute(batch_count_stmt).scalar() or 0

    # Create a new batch
    new_batch_name = batch_name or f"{project.name}-batch-{batch_count + 1}"
    new_batch = Batch(
        organization_id=organization.id,
        user_id=current_user["user_id"],
        project_id=project_id,
        name=new_batch_name,
        description=f"Batch {batch_count + 1} for project {project.name}",
        quality_summary={},
    )

    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    # Create batch folder
    try:
        storage_backend = get_storage_backend(project)
        batch_path = f"projects/{project.id}/batches/{new_batch.id}"
        
        # Handle different storage backends
        if hasattr(storage_backend, 'base_dir'):
            # Local storage backend
            os.makedirs(os.path.join(storage_backend.base_dir, batch_path), exist_ok=True)
        else:
            # S3 storage backend - create empty object to represent folder
            try:
                # For S3, we create an empty object with trailing slash to represent folder
                storage_backend.s3_client.put_object(
                    Bucket=storage_backend.bucket_name,
                    Key=f"{batch_path}/"
                )
            except Exception as s3_error:
                print(f"Error creating S3 folder: {str(s3_error)}")
    except Exception as e:
        # Log the error but don't fail the batch creation
        print(f"Error creating batch folder: {str(e)}")

    return new_batch


@router.get("/{batch_id}/quality-summary")
async def get_batch_quality_summary(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get quality summary for a batch"""
    # First check if the batch exists and belongs to the user and organization
    batch_stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.organization_id == organization.id,
        Batch.user_id == current_user["user_id"],
    )

    batch_result = db.execute(batch_stmt)
    batch = batch_result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all datasets for this batch
    datasets_stmt = select(SyntheticDataset).where(
        SyntheticDataset.batch_id == batch_id,
        SyntheticDataset.organization_id == organization.id,
        SyntheticDataset.user_id == current_user["user_id"],
    )

    datasets_result = db.execute(datasets_stmt)
    datasets = datasets_result.scalars().all()

    # Aggregate quality metrics from all datasets
    quality_summary = {
        "total_datasets": len(datasets),
        "total_checks": 0,
        "checks_by_type": {},
        "issues_by_severity": {"info": 0, "warning": 0, "error": 0},
    }

    for dataset in datasets:
        if dataset.check_results:
            # Count total checks
            for column, checks in dataset.check_results.items():
                quality_summary["total_checks"] += len(checks)

                # Count checks by type
                for check_name in checks.keys():
                    if check_name not in quality_summary["checks_by_type"]:
                        quality_summary["checks_by_type"][check_name] = 0
                    quality_summary["checks_by_type"][check_name] += 1

            # Count issues by severity (this would require additional logic based on check results)
            # For now, we'll just use placeholder logic
            for column, checks in dataset.check_results.items():
                for check_name, results in checks.items():
                    if "error" in results:
                        quality_summary["issues_by_severity"]["error"] += 1
                    elif (
                        "out_of_range_count" in results
                        and results["out_of_range_count"] > 0
                    ):
                        quality_summary["issues_by_severity"]["warning"] += 1
                    elif "missing_count" in results and results["missing_count"] > 0:
                        quality_summary["issues_by_severity"]["info"] += 1

    # Update the batch's quality summary
    batch.quality_summary = quality_summary
    db.commit()

    return quality_summary
