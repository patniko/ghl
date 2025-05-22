"""
Batch processing service for handling batch processing operations.
"""

import os
import shutil
import asyncio
import logging
from typing import Dict, Optional, List
from fastapi import HTTPException

from models import Batch, Project, BatchProcessingStatus
from services.storage import get_storage_backend, get_project_storage_path
from db import get_db

from loguru import logger

# Processing status constants
PROCESSING_STATUS = {
    "QUEUED": "queued",
    "PROCESSING": "processing",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled"
}

# Dictionary to track running processing tasks
# Key: batch_id, Value: Dict with task info and asyncio.Task
running_tasks: Dict[int, Dict] = {}


async def process_batch(batch_id: int, organization_id: int, user_id: int) -> Dict:
    """
    Process a batch by copying files to a processing directory or downloading from S3.
    
    Args:
        batch_id: The ID of the batch to process
        organization_id: The organization ID
        user_id: The user ID
        
    Returns:
        Dict with processing status and details
    """
    db = next(get_db())
    
    try:
        # Get the batch
        batch = db.query(Batch).filter(
            Batch.id == batch_id,
            Batch.organization_id == organization_id,
            Batch.user_id == user_id
        ).first()
        
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Get the project
        project = None
        if batch.project_id:
            project = db.query(Project).filter(Project.id == batch.project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        
        # Update batch status to processing
        batch.processing_status = PROCESSING_STATUS["PROCESSING"]
        db.commit()
        
        # Get storage backend
        storage_backend = get_storage_backend(project)
        
        # Create processing directory
        processing_dir = f"processing/{batch.id}"
        
        # Handle different storage backends
        if hasattr(storage_backend, 'base_dir'):
            # Local storage backend
            source_dir = f"projects/{project.id}/batches/{batch.id}" if project else f"batches/{batch.id}"
            full_source_dir = os.path.join(storage_backend.base_dir, source_dir)
            full_processing_dir = os.path.join(storage_backend.base_dir, processing_dir)
            
            # Create processing directory
            os.makedirs(full_processing_dir, exist_ok=True)
            
            # List files in the batch directory
            files = await storage_backend.list_files(source_dir)
            
            # Copy each file to the processing directory
            for file_path in files:
                source_path = os.path.join(storage_backend.base_dir, file_path)
                dest_path = os.path.join(full_processing_dir, os.path.basename(file_path))
                shutil.copy2(source_path, dest_path)
                
                # Simulate processing time
                await asyncio.sleep(0.5)
                
                # Check if task was cancelled
                if batch_id in running_tasks and running_tasks[batch_id].get("cancelled", False):
                    # Clean up processing directory
                    if os.path.exists(full_processing_dir):
                        shutil.rmtree(full_processing_dir)
                    
                    # Update batch status to cancelled
                    batch.processing_status = PROCESSING_STATUS["CANCELLED"]
                    db.commit()
                    
                    # Remove the task from running_tasks when cancelled
                    if batch_id in running_tasks:
                        del running_tasks[batch_id]
                    
                    return {
                        "status": PROCESSING_STATUS["CANCELLED"],
                        "message": "Batch processing was cancelled",
                        "batch_id": batch_id
                    }
        else:
            # S3 storage backend
            source_prefix = f"projects/{project.id}/batches/{batch.id}/" if project else f"batches/{batch.id}/"
            
            # List files in the S3 bucket with the specified prefix
            files = await storage_backend.list_files(source_prefix)
            
            # Create a local temporary directory for processing
            local_processing_dir = f"uploads/processing/{batch.id}"
            os.makedirs(local_processing_dir, exist_ok=True)
            
            # Download each file from S3 to the local processing directory
            for file_path in files:
                # Skip directory markers (objects ending with '/')
                if file_path.endswith('/'):
                    continue
                
                # Get the file from S3
                s3_path = f"s3://{storage_backend.bucket_name}/{file_path}"
                file_obj = await storage_backend.get_file(s3_path)
                
                # Save the file to the local processing directory
                local_file_path = os.path.join(local_processing_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_obj.read())
                
                # Simulate processing time
                await asyncio.sleep(0.5)
                
                # Check if task was cancelled
                if batch_id in running_tasks and running_tasks[batch_id].get("cancelled", False):
                    # Clean up processing directory
                    if os.path.exists(local_processing_dir):
                        shutil.rmtree(local_processing_dir)
                    
                    # Update batch status to cancelled
                    batch.processing_status = PROCESSING_STATUS["CANCELLED"]
                    db.commit()
                    
                    # Remove the task from running_tasks when cancelled
                    if batch_id in running_tasks:
                        del running_tasks[batch_id]
                    
                    return {
                        "status": PROCESSING_STATUS["CANCELLED"],
                        "message": "Batch processing was cancelled",
                        "batch_id": batch_id
                    }
        
        # Update batch status to completed
        batch.processing_status = PROCESSING_STATUS["COMPLETED"]
        db.commit()
        
        # Remove the task from running_tasks when completed
        if batch_id in running_tasks:
            del running_tasks[batch_id]
        
        return {
            "status": PROCESSING_STATUS["COMPLETED"],
            "message": "Batch processing completed successfully",
            "batch_id": batch_id
        }
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")
        
        # Update batch status to failed
        try:
            batch = db.query(Batch).filter(Batch.id == batch_id).first()
            if batch:
                batch.processing_status = PROCESSING_STATUS["FAILED"]
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating batch status: {str(db_error)}")
        
        # Remove the task from running_tasks when failed
        if batch_id in running_tasks:
            del running_tasks[batch_id]
        
        return {
            "status": PROCESSING_STATUS["FAILED"],
            "message": f"Batch processing failed: {str(e)}",
            "batch_id": batch_id
        }


async def start_batch_processing(batch_id: int, organization_id: int, user_id: int) -> Dict:
    """
    Start processing a batch asynchronously.
    
    Args:
        batch_id: The ID of the batch to process
        organization_id: The organization ID
        user_id: The user ID
        
    Returns:
        Dict with processing status and details
    """
    # Check if batch is already being processed
    if batch_id in running_tasks and not running_tasks[batch_id].get("cancelled", False):
        return {
            "status": "already_processing",
            "message": "Batch is already being processed",
            "batch_id": batch_id
        }
    
    # Create and start the processing task
    task = asyncio.create_task(process_batch(batch_id, organization_id, user_id))
    
    # Store the task in the running_tasks dictionary
    running_tasks[batch_id] = {
        "task": task,
        "organization_id": organization_id,
        "user_id": user_id,
        "cancelled": False
    }
    
    # Update batch status to queued
    db = next(get_db())
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if batch:
        batch.processing_status = PROCESSING_STATUS["QUEUED"]
        db.commit()
    
    return {
        "status": PROCESSING_STATUS["QUEUED"],
        "message": "Batch processing started",
        "batch_id": batch_id
    }


async def cancel_batch_processing(batch_id: int) -> Dict:
    """
    Cancel a running batch processing task.
    
    Args:
        batch_id: The ID of the batch to cancel
        
    Returns:
        Dict with cancellation status and details
    """
    if batch_id not in running_tasks:
        return {
            "status": "not_processing",
            "message": "Batch is not being processed",
            "batch_id": batch_id
        }
    
    # Mark the task as cancelled
    running_tasks[batch_id]["cancelled"] = True
    
    return {
        "status": "cancelling",
        "message": "Batch processing cancellation requested",
        "batch_id": batch_id
    }


async def get_batch_processing_status(batch_id: int) -> Dict:
    """
    Get the current processing status of a batch.
    
    Args:
        batch_id: The ID of the batch
        
    Returns:
        Dict with processing status and details
    """
    db = next(get_db())
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    
    if not batch:
        return {
            "status": "not_found",
            "message": "Batch not found",
            "batch_id": batch_id
        }
    
    # Check if batch is being processed
    is_processing = batch_id in running_tasks and not running_tasks[batch_id].get("cancelled", False)
    
    return {
        "status": batch.processing_status or "unknown",
        "is_processing": is_processing,
        "batch_id": batch_id
    }
