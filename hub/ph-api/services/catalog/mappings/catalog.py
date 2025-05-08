from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from middleware import validate_user_organization
from models import (
    User,
    Organization,
    ColumnMapping,
    ColumnMappingCreate,
    ColumnMappingResponse,
    ColumnMappingUpdate,
    DataType,
)

router = APIRouter()


# CRUD operations for column mappings
@router.post("/", response_model=ColumnMappingResponse)
async def create_column_mapping(
    mapping: ColumnMappingCreate,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(validate_user_organization),
    db: Session = Depends(get_db),
):
    """Create a new column mapping"""
    new_mapping = ColumnMapping(
        organization_id=organization.id,  # Set organization_id from middleware
        user_id=current_user["user_id"],
        column_name=mapping.column_name,
        data_type=mapping.data_type.value,  # Use string value instead of enum
        description=mapping.description,
        synonyms=mapping.synonyms,
    )

    db.add(new_mapping)
    db.commit()
    db.refresh(new_mapping)

    return new_mapping


@router.get("/", response_model=List[ColumnMappingResponse])
async def get_column_mappings(
    data_type: Optional[DataType] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all column mappings for the current user with optional filters"""
    query = select(ColumnMapping).where(
        ColumnMapping.user_id == current_user["user_id"]
    )

    # Apply filters if provided
    if data_type:
        query = query.where(
            ColumnMapping.data_type == data_type.value
        )  # Use string value

    result = db.execute(query)
    mappings = result.scalars().all()

    return mappings


@router.get("/{mapping_id}", response_model=ColumnMappingResponse)
async def get_column_mapping(
    mapping_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific column mapping"""
    stmt = select(ColumnMapping).where(
        ColumnMapping.id == mapping_id, ColumnMapping.user_id == current_user["user_id"]
    )
    result = db.execute(stmt)
    mapping = result.scalar_one_or_none()

    if not mapping:
        raise HTTPException(status_code=404, detail="Column mapping not found")

    return mapping


@router.put("/{mapping_id}", response_model=ColumnMappingResponse)
async def update_column_mapping(
    mapping_id: int,
    mapping_update: ColumnMappingUpdate,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Update a column mapping"""
    stmt = select(ColumnMapping).where(
        ColumnMapping.id == mapping_id, ColumnMapping.user_id == current_user["user_id"]
    )
    result = db.execute(stmt)
    mapping = result.scalar_one_or_none()

    if not mapping:
        raise HTTPException(status_code=404, detail="Column mapping not found")

    # Update fields if provided
    if mapping_update.data_type is not None:
        mapping.data_type = mapping_update.data_type.value  # Use string value

    if mapping_update.description is not None:
        mapping.description = mapping_update.description

    if mapping_update.synonyms is not None:
        mapping.synonyms = mapping_update.synonyms

    db.commit()
    db.refresh(mapping)

    return mapping


@router.delete("/{mapping_id}")
async def delete_column_mapping(
    mapping_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a column mapping"""
    stmt = select(ColumnMapping).where(
        ColumnMapping.id == mapping_id, ColumnMapping.user_id == current_user["user_id"]
    )
    result = db.execute(stmt)
    mapping = result.scalar_one_or_none()

    if not mapping:
        raise HTTPException(status_code=404, detail="Column mapping not found")

    db.delete(mapping)
    db.commit()

    return {"message": "Column mapping deleted successfully"}
