"""
Batch processing API endpoints for handling batch processing operations.
"""

from typing import Dict
from fastapi import APIRouter, Depends, HTTPException

from auth import validate_jwt
from db import get_db
from middleware import validate_user_organization
from models import User, Organization, Batch, BatchProcessingStatus
from services.batch_processor import start_batch_processing, cancel_batch_processing, get_batch_processing_status

router = APIRouter()


@router.post("/process")
async def process_batch(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
):
    """
    Start processing a batch.
    
    Args:
        batch_id: The ID of the batch to process
        organization: The organization (from dependency)
        current_user: The current user (from dependency)
        
    Returns:
        Dict with processing status and details
    """
    # Start batch processing
    result = await start_batch_processing(
        batch_id=batch_id,
        organization_id=organization.id,
        user_id=current_user["user_id"]
    )
    
    return result


@router.post("/cancel")
async def cancel_batch_processing_endpoint(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
):
    """
    Cancel a running batch processing task.
    
    Args:
        batch_id: The ID of the batch to cancel
        organization: The organization (from dependency)
        current_user: The current user (from dependency)
        
    Returns:
        Dict with cancellation status and details
    """
    # Cancel batch processing
    result = await cancel_batch_processing(batch_id)
    
    return result


@router.get("/status")
async def get_batch_processing_status_endpoint(
    batch_id: int,
    organization: Organization = Depends(validate_user_organization),
    current_user: User = Depends(validate_jwt),
):
    """
    Get the current processing status of a batch.
    
    Args:
        batch_id: The ID of the batch
        organization: The organization (from dependency)
        current_user: The current user (from dependency)
        
    Returns:
        Dict with processing status and details
    """
    # Get batch processing status
    result = await get_batch_processing_status(batch_id)
    
    return result
