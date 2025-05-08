from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from middleware import get_organization_from_path
from models import (
    User,
    Model,
    ModelCreate,
    ModelResponse,
    ModelUpdate,
    ModelEvaluationResponse,
    File,
    FileType,
)

router = APIRouter()

# Dictionary mapping model implementation names to functions
MODEL_IMPLEMENTATIONS = {
    "echo_quality": "Analyze the quality of echo images.",
    "text_classification": "Classify text into predefined categories",
    "image_classification": "Classify images into predefined categories",
    "object_detection": "Detect and locate objects in images",
    "sentiment_analysis": "Analyze sentiment in text (positive, negative, neutral)",
    "named_entity_recognition": "Identify and classify named entities in text",
    "text_summarization": "Generate concise summaries of longer text",
    "translation": "Translate text from one language to another",
    "question_answering": "Answer questions based on provided context",
    "speech_recognition": "Convert spoken language to text",
    "anomaly_detection": "Identify unusual patterns in data",
}


# CRUD operations for models
@router.post("/", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Create a new model"""
    # Validate that the implementation exists
    if model.implementation not in MODEL_IMPLEMENTATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Implementation '{model.implementation}' not found. Available implementations: {list(MODEL_IMPLEMENTATIONS.keys())}",
        )

    # Create the model
    new_model = Model(
        organization_id=organization.id,
        name=model.name,
        description=model.description,
        version=model.version,
        parameters=model.parameters,
        implementation=model.implementation,
        is_system=False,  # User-created models are not system models
    )

    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    return new_model


@router.get("/", response_model=List[ModelResponse])
async def get_models(
    implementation: Optional[str] = None,
    is_system: Optional[bool] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get all models with optional filters"""
    query = select(Model).where(Model.organization_id == organization.id)

    # Apply filters if provided
    if implementation:
        query = query.where(Model.implementation == implementation)

    if is_system is not None:
        query = query.where(Model.is_system == is_system)

    result = db.execute(query)
    models = result.scalars().all()

    return models


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get a specific model"""
    stmt = select(Model).where(
        Model.id == model_id, Model.organization_id == organization.id
    )
    result = db.execute(stmt)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelUpdate,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Update a model"""
    stmt = select(Model).where(
        Model.id == model_id, Model.organization_id == organization.id
    )
    result = db.execute(stmt)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Prevent updating system models
    if model.is_system:
        raise HTTPException(status_code=403, detail="System models cannot be modified")

    # Update fields if provided
    if model_update.name is not None:
        model.name = model_update.name

    if model_update.description is not None:
        model.description = model_update.description

    if model_update.version is not None:
        model.version = model_update.version

    if model_update.parameters is not None:
        model.parameters = model_update.parameters

    if model_update.implementation is not None:
        # Validate that the implementation exists
        if model_update.implementation not in MODEL_IMPLEMENTATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Implementation '{model_update.implementation}' not found. Available implementations: {list(MODEL_IMPLEMENTATIONS.keys())}",
            )
        model.implementation = model_update.implementation

    # Allow toggling the system flag if provided
    if hasattr(model_update, "is_system") and model_update.is_system is not None:
        model.is_system = model_update.is_system

    db.commit()
    db.refresh(model)

    return model


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Delete a model"""
    stmt = select(Model).where(
        Model.id == model_id, Model.organization_id == organization.id
    )
    result = db.execute(stmt)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Prevent deleting system models
    if model.is_system:
        raise HTTPException(status_code=403, detail="System models cannot be deleted")

    db.delete(model)
    db.commit()

    return {"message": "Model deleted successfully"}


@router.post("/evaluate-file/{file_id}", response_model=ModelEvaluationResponse)
async def evaluate_file(
    file_id: int,
    model_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Evaluate a file using a specific model"""
    # Get the file
    file_stmt = select(File).where(
        File.id == file_id,
        File.user_id == current_user["user_id"],
        File.organization_id == organization.id,
    )
    file_result = db.execute(file_stmt)
    file = file_result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # Get the model
    model_stmt = select(Model).where(
        Model.id == model_id, Model.organization_id == organization.id
    )
    model_result = db.execute(model_stmt)
    model = model_result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if the file type is compatible with the model
    # This is a simplified check - in a real implementation, you would have more sophisticated compatibility logic
    compatible = True
    incompatibility_reason = None

    # Example compatibility check
    if model.implementation == "image_classification" and file.file_type not in [
        FileType.DICOM,
        "jpg",
        "png",
        "jpeg",
    ]:
        compatible = False
        incompatibility_reason = (
            f"Model '{model.name}' requires image files, but got {file.file_type}"
        )
    elif model.implementation == "text_classification" and file.file_type not in [
        FileType.CSV,
        "txt",
        "json",
    ]:
        compatible = False
        incompatibility_reason = (
            f"Model '{model.name}' requires text files, but got {file.file_type}"
        )

    if not compatible:
        return ModelEvaluationResponse(
            compatible=False,
            reason=incompatibility_reason,
            file_id=file_id,
            model_id=model_id,
        )

    # In a real implementation, you would actually run the model on the file here
    # For this example, we'll just return a mock result
    mock_result = {
        "model_name": model.name,
        "model_version": model.version,
        "file_name": file.original_filename,
        "file_type": file.file_type,
        "evaluation_time": "2025-03-18T19:22:00Z",
        "results": {
            "confidence": 0.95,
            "predictions": [
                {"label": "Class A", "probability": 0.8},
                {"label": "Class B", "probability": 0.15},
                {"label": "Class C", "probability": 0.05},
            ],
        },
    }

    # Update file metadata with model evaluation results
    if not file.file_metadata:
        file.file_metadata = {}

    if "model_evaluations" not in file.file_metadata:
        file.file_metadata["model_evaluations"] = []

    file.file_metadata["model_evaluations"].append(
        {
            "model_id": model.id,
            "model_name": model.name,
            "evaluation_time": mock_result["evaluation_time"],
            "results": mock_result["results"],
        }
    )

    db.commit()

    return ModelEvaluationResponse(
        compatible=True,
        file_id=file_id,
        model_id=model_id,
        evaluation_results=mock_result,
    )
