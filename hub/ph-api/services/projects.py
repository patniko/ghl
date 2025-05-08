"""
Projects service for managing projects and their storage settings.
"""

import re
import os
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from models import (
    User,
    Project,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    DataRegion,
    Batch,
)
from middleware import get_organization_from_path
from services.storage import get_storage_backend

router = APIRouter()


def create_project(
    project: ProjectCreate,
    current_user: Dict[str, Any],
    db: Session,
    organization: Any,
) -> Project:
    """
    Create a new project.

    Args:
        project: Project data
        current_user: Current user information
        db: Database session
        organization: Organization object

    Returns:
        Created project

    Raises:
        HTTPException: If data_region is invalid or if project name contains invalid characters
    """
    # Validate data_region if provided
    if project.data_region and project.data_region not in [r.value for r in DataRegion]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_region. Must be one of: {', '.join([r.value for r in DataRegion])}",
        )

    # Validate project name (should only contain alphanumeric characters and hyphens)
    if not re.match(r"^[a-zA-Z0-9-]+$", project.name):
        raise HTTPException(
            status_code=400,
            detail="Project name must contain only alphanumeric characters and hyphens",
        )

    # Check if a project with the same name already exists in this organization
    existing_project = (
        db.query(Project)
        .filter(
            Project.organization_id == organization.id, Project.name == project.name
        )
        .first()
    )

    if existing_project:
        raise HTTPException(
            status_code=400,
            detail=f"A project with the name '{project.name}' already exists in this organization",
        )

    # Create project
    db_project = Project(
        organization_id=organization.id,
        user_id=current_user["user_id"],
        name=project.name,
        description=project.description,
        data_region=project.data_region,
        s3_bucket_name=project.s3_bucket_name,
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    # Create initial batch for the project
    initial_batch = Batch(
        organization_id=organization.id,
        user_id=current_user["user_id"],
        project_id=db_project.id,
        name=f"{project.name}-batch-1",
        description="Initial batch created automatically",
        quality_summary={},
    )
    db.add(initial_batch)
    db.commit()
    db.refresh(initial_batch)

    # Update project with first batch ID
    db_project.first_batch_id = initial_batch.id
    db.commit()
    db.refresh(db_project)

    # Create storage folder structure for the project
    try:
        storage_backend = get_storage_backend(db_project)

        # Create project folder
        project_path = f"projects/{db_project.id}"
        batch_path = f"projects/{db_project.id}/batches/{initial_batch.id}"

        # Handle different storage backends
        if hasattr(storage_backend, 'base_dir'):
            # Local storage backend
            os.makedirs(os.path.join(storage_backend.base_dir, project_path), exist_ok=True)
            os.makedirs(os.path.join(storage_backend.base_dir, batch_path), exist_ok=True)
        else:
            # S3 storage backend - create empty objects to represent folders
            try:
                # For S3, we create empty objects with trailing slashes to represent folders
                storage_backend.s3_client.put_object(
                    Bucket=storage_backend.bucket_name,
                    Key=f"{project_path}/"
                )
                storage_backend.s3_client.put_object(
                    Bucket=storage_backend.bucket_name,
                    Key=f"{batch_path}/"
                )
            except Exception as s3_error:
                print(f"Error creating S3 folders: {str(s3_error)}")
    except Exception as e:
        # Log the error but don't fail the project creation
        print(f"Error creating project folders: {str(e)}")

    return db_project


def get_projects(
    current_user: Dict[str, Any],
    db: Session,
    organization: Any,
) -> List[Project]:
    """
    Get all projects for the current user in the organization.

    Args:
        current_user: Current user information
        db: Database session
        organization: Organization object

    Returns:
        List of projects
    """
    stmt = select(Project).where(
        Project.user_id == current_user["user_id"],
        Project.organization_id == organization.id,
    )
    result = db.execute(stmt)
    projects = result.scalars().all()
    return projects


def get_project(
    project_name: str,
    current_user: Dict[str, Any],
    db: Session,
    organization: Any,
) -> Project:
    """
    Get a specific project.

    Args:
        project_name: Project name
        current_user: Current user information
        db: Database session
        organization: Organization object

    Returns:
        Project object

    Raises:
        HTTPException: If project not found
    """
    stmt = select(Project).where(
        Project.name == project_name,
        Project.user_id == current_user["user_id"],
        Project.organization_id == organization.id,
    )
    result = db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def update_project(
    project_name: str,
    project_update: ProjectUpdate,
    current_user: Dict[str, Any],
    db: Session,
    organization: Any,
) -> Project:
    """
    Update a project.

    Args:
        project_name: Project name
        project_update: Project update data
        current_user: Current user information
        db: Database session
        organization: Organization object

    Returns:
        Updated project

    Raises:
        HTTPException: If project not found or data_region is invalid
    """
    # Get the project
    stmt = select(Project).where(
        Project.name == project_name,
        Project.user_id == current_user["user_id"],
        Project.organization_id == organization.id,
    )
    result = db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate data_region if provided
    if project_update.data_region and project_update.data_region not in [
        r.value for r in DataRegion
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_region. Must be one of: {', '.join([r.value for r in DataRegion])}",
        )

    # Validate project name if provided
    if project_update.name is not None:
        # Validate project name format
        if not re.match(r"^[a-zA-Z0-9-]+$", project_update.name):
            raise HTTPException(
                status_code=400,
                detail="Project name must contain only alphanumeric characters and hyphens",
            )

    # Check if another project with the same name already exists in this organization
    if project_update.name is not None:
        existing_project = (
            db.query(Project)
            .filter(
                Project.organization_id == organization.id,
                Project.name == project_update.name,
                Project.name != project_name,  # Exclude current project
            )
            .first()
        )

        if existing_project:
            raise HTTPException(
                status_code=400,
                detail=f"A project with the name '{project_update.name}' already exists in this organization",
            )

        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description
    if project_update.data_region is not None:
        project.data_region = project_update.data_region
    if project_update.s3_bucket_name is not None:
        project.s3_bucket_name = project_update.s3_bucket_name

    db.commit()
    db.refresh(project)
    return project


def delete_project(
    project_name: str,
    current_user: Dict[str, Any],
    db: Session,
    organization: Any,
) -> Dict[str, str]:
    """
    Delete a project.

    Args:
        project_name: Project name
        current_user: Current user information
        db: Database session
        organization: Organization object

    Returns:
        Success message

    Raises:
        HTTPException: If project not found
    """
    # Get the project
    stmt = select(Project).where(
        Project.name == project_name,
        Project.user_id == current_user["user_id"],
        Project.organization_id == organization.id,
    )
    result = db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete the project
    db.delete(project)
    db.commit()
    return {"message": "Project deleted successfully"}


@router.post("", response_model=ProjectResponse)
async def api_create_project(
    project_data: dict,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Create a new project"""
    # Check for invalid data_region before creating ProjectCreate instance
    if "data_region" in project_data and project_data["data_region"] not in [
        r.value for r in DataRegion
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_region. Must be one of: {', '.join([r.value for r in DataRegion])}",
        )

    try:
        project = ProjectCreate(**project_data)
        return create_project(project, current_user, db, organization)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[ProjectResponse])
async def api_get_projects(
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get all projects for the current user in the organization"""
    return get_projects(current_user, db, organization)


@router.get("/{project_name}", response_model=ProjectResponse)
async def api_get_project(
    project_name: str,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get a specific project"""
    return get_project(project_name, current_user, db, organization)


@router.put("/{project_name}", response_model=ProjectResponse)
async def api_update_project(
    project_name: str,
    project_data: dict,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Update a project"""
    # Check for invalid data_region before creating ProjectUpdate instance
    if "data_region" in project_data and project_data["data_region"] not in [
        r.value for r in DataRegion
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_region. Must be one of: {', '.join([r.value for r in DataRegion])}",
        )

    try:
        project_update = ProjectUpdate(**project_data)
        return update_project(
            project_name, project_update, current_user, db, organization
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{project_name}")
async def api_delete_project(
    project_name: str,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Delete a project"""
    return delete_project(project_name, current_user, db, organization)
