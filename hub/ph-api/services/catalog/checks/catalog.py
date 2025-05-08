from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from auth import validate_jwt
from db import get_db
from middleware import validate_user_organization
from models import (
    CheckScope,
    User,
    Organization,
    Check,
    CheckCreate,
    CheckResponse,
    CheckUpdate,
    ColumnMapping,
    DataType,
    SyntheticDataset,
)

router = APIRouter()


# Check implementations
def check_range_compliance(
    values: List[float], min_val: float, max_val: float
) -> Dict[str, Any]:
    """Check if values comply with a specified range"""
    if not values or len(values) == 0:
        return {
            "in_range_count": 0,
            "out_of_range_count": 0,
            "compliance_percentage": 0,
        }

    in_range = [val for val in values if min_val <= val <= max_val]
    out_of_range = [val for val in values if val < min_val or val > max_val]

    compliance_percentage = (len(in_range) / len(values)) * 100

    return {
        "in_range_count": len(in_range),
        "out_of_range_count": len(out_of_range),
        "compliance_percentage": float(compliance_percentage),
    }


def check_missing_values(values: List[Any]) -> Dict[str, Any]:
    """Check for missing values in a column"""
    if not values:
        return {"total_count": 0, "missing_count": 0, "missing_percentage": 100.0}

    # Count None, empty strings, and NaN values as missing
    missing_count = sum(
        1
        for val in values
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val))
    )
    total_count = len(values)
    missing_percentage = (missing_count / total_count) * 100

    return {
        "total_count": total_count,
        "missing_count": missing_count,
        "missing_percentage": float(missing_percentage),
    }


def check_outliers(values: List[float]) -> Dict[str, Any]:
    """Check for outliers using IQR method"""
    if not values or len(values) < 4:  # Need at least 4 values for meaningful quartiles
        return {
            "total_count": len(values) if values else 0,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "outliers": [],
        }

    # Filter out None and NaN values
    filtered_values = [
        val
        for val in values
        if val is not None and not (isinstance(val, float) and np.isnan(val))
    ]

    if len(filtered_values) < 4:
        return {
            "total_count": len(values),
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "outliers": [],
        }

    # Calculate quartiles and IQR
    q1 = np.percentile(filtered_values, 25)
    q3 = np.percentile(filtered_values, 75)
    iqr = q3 - q1

    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = [
        val for val in filtered_values if val < lower_bound or val > upper_bound
    ]
    outlier_percentage = (len(outliers) / len(filtered_values)) * 100

    return {
        "total_count": len(filtered_values),
        "outlier_count": len(outliers),
        "outlier_percentage": float(outlier_percentage),
        "outliers": outliers[
            :10
        ],  # Return only first 10 outliers to avoid large responses
    }


def check_categorical_distribution(values: List[str]) -> Dict[str, Any]:
    """Check distribution of categorical values"""
    if not values:
        return {"total_count": 0, "unique_count": 0, "distribution": {}}

    # Filter out None and empty values
    filtered_values = [val for val in values if val is not None and val != ""]

    if not filtered_values:
        return {"total_count": len(values), "unique_count": 0, "distribution": {}}

    # Count occurrences of each value
    value_counts = {}
    for val in filtered_values:
        value_counts[val] = value_counts.get(val, 0) + 1

    # Calculate percentages
    total = len(filtered_values)
    distribution = {
        val: {"count": count, "percentage": (count / total) * 100}
        for val, count in value_counts.items()
    }

    return {
        "total_count": total,
        "unique_count": len(value_counts),
        "distribution": distribution,
    }


def check_numeric_statistics(values: List[float]) -> Dict[str, Any]:
    """Calculate basic statistics for numeric values"""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
        }

    # Filter out None and NaN values
    filtered_values = [
        val
        for val in values
        if val is not None and not (isinstance(val, float) and np.isnan(val))
    ]

    if not filtered_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
        }

    # Calculate statistics
    return {
        "count": len(filtered_values),
        "mean": float(np.mean(filtered_values)),
        "median": float(np.median(filtered_values)),
        "std_dev": float(np.std(filtered_values)),
        "min": float(np.min(filtered_values)),
        "max": float(np.max(filtered_values)),
    }


def check_date_range(values: List[str]) -> Dict[str, Any]:
    """Check range of dates"""
    if not values:
        return {"count": 0, "min_date": None, "max_date": None, "range_days": None}

    # Filter out None and empty values
    filtered_values = [val for val in values if val is not None and val != ""]

    if not filtered_values:
        return {"count": 0, "min_date": None, "max_date": None, "range_days": None}

    try:
        # Convert to pandas datetime
        dates = pd.to_datetime(filtered_values)

        # Calculate min and max dates
        min_date = dates.min()
        max_date = dates.max()

        # Calculate range in days
        range_days = (max_date - min_date).days

        return {
            "count": len(filtered_values),
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "range_days": range_days,
        }
    except Exception as e:
        return {
            "count": len(filtered_values),
            "error": str(e),
            "min_date": None,
            "max_date": None,
            "range_days": None,
        }


def check_text_length(values: List[str]) -> Dict[str, Any]:
    """Check length statistics for text values"""
    if not values:
        return {"count": 0, "mean_length": None, "min_length": None, "max_length": None}

    # Filter out None values
    filtered_values = [val for val in values if val is not None]

    if not filtered_values:
        return {"count": 0, "mean_length": None, "min_length": None, "max_length": None}

    # Calculate lengths
    lengths = [len(str(val)) for val in filtered_values]

    return {
        "count": len(filtered_values),
        "mean_length": float(np.mean(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def check_boolean_distribution(values: List[bool]) -> Dict[str, Any]:
    """Check distribution of boolean values"""
    if not values:
        return {
            "count": 0,
            "true_count": 0,
            "false_count": 0,
            "true_percentage": None,
            "false_percentage": None,
        }

    # Filter out None values
    filtered_values = [val for val in values if val is not None]

    if not filtered_values:
        return {
            "count": 0,
            "true_count": 0,
            "false_count": 0,
            "true_percentage": None,
            "false_percentage": None,
        }

    # Count true and false values
    true_count = sum(1 for val in filtered_values if val)
    false_count = len(filtered_values) - true_count

    return {
        "count": len(filtered_values),
        "true_count": true_count,
        "false_count": false_count,
        "true_percentage": (true_count / len(filtered_values)) * 100,
        "false_percentage": (false_count / len(filtered_values)) * 100,
    }


# Dictionary mapping check implementation names to functions
CHECK_IMPLEMENTATIONS = {
    "check_range_compliance": check_range_compliance,
    "check_missing_values": check_missing_values,
    "check_outliers": check_outliers,
    "check_categorical_distribution": check_categorical_distribution,
    "check_numeric_statistics": check_numeric_statistics,
    "check_date_range": check_date_range,
    "check_text_length": check_text_length,
    "check_boolean_distribution": check_boolean_distribution,
}


# CRUD operations for checks
# Python script validation
def validate_python_script(script: str) -> tuple[bool, str]:
    """Validate a Python script for syntax errors"""
    if not script:
        return True, ""

    try:
        compile(script, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


@router.post("/", response_model=CheckResponse)
async def create_check(
    check: CheckCreate,
    current_user: User = Depends(validate_jwt),
    organization: Organization = Depends(validate_user_organization),
    db: Session = Depends(get_db),
):
    """Create a new check"""
    # Validate that the implementation exists
    if check.implementation not in CHECK_IMPLEMENTATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Implementation '{check.implementation}' not found. Available implementations: {list(CHECK_IMPLEMENTATIONS.keys())}",
        )

    # Validate Python script if provided
    if check.python_script and (
        check.scope == CheckScope.ROW or check.scope == CheckScope.FILE
    ):
        is_valid, error_msg = validate_python_script(check.python_script)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Python script: {error_msg}",
            )

    # Create the check
    new_check = Check(
        organization_id=organization.id,  # Set organization_id from middleware
        name=check.name,
        description=check.description,
        data_type=check.data_type.value,  # Use string value instead of enum
        scope=check.scope.value,  # Use string value instead of enum
        parameters=check.parameters,
        implementation=check.implementation,
        python_script=check.python_script,
        is_system=False,  # User-created checks are not system checks
    )

    db.add(new_check)
    db.commit()
    db.refresh(new_check)

    return new_check


@router.get("/", response_model=List[CheckResponse])
async def get_checks(
    data_type: Optional[DataType] = None,
    is_system: Optional[bool] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all checks with optional filters"""
    query = select(Check)

    # Apply filters if provided
    if data_type:
        query = query.where(Check.data_type == data_type.value)  # Use string value

    if is_system is not None:
        query = query.where(Check.is_system == is_system)

    result = db.execute(query)
    checks = result.scalars().all()

    # Ensure python_script is always a string, not None
    for check in checks:
        if check.python_script is None:
            check.python_script = ""

    return checks


@router.get("/{check_id}", response_model=CheckResponse)
async def get_check(
    check_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific check"""
    stmt = select(Check).where(Check.id == check_id)
    result = db.execute(stmt)
    check = result.scalar_one_or_none()

    if not check:
        raise HTTPException(status_code=404, detail="Check not found")

    # Ensure python_script is always a string, not None
    if check.python_script is None:
        check.python_script = ""

    return check


@router.put("/{check_id}", response_model=CheckResponse)
async def update_check(
    check_id: int,
    check_update: CheckUpdate,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Update a check"""
    stmt = select(Check).where(Check.id == check_id)
    result = db.execute(stmt)
    check = result.scalar_one_or_none()

    if not check:
        raise HTTPException(status_code=404, detail="Check not found")

    # Prevent updating system checks
    if check.is_system:
        raise HTTPException(status_code=403, detail="System checks cannot be modified")

    # Update fields if provided
    if check_update.name is not None:
        check.name = check_update.name

    if check_update.description is not None:
        check.description = check_update.description

    if check_update.scope is not None:
        check.scope = check_update.scope.value

    if check_update.parameters is not None:
        check.parameters = check_update.parameters

    if check_update.implementation is not None:
        # Validate that the implementation exists
        if check_update.implementation not in CHECK_IMPLEMENTATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Implementation '{check_update.implementation}' not found. Available implementations: {list(CHECK_IMPLEMENTATIONS.keys())}",
            )
        check.implementation = check_update.implementation

    if check_update.python_script is not None:
        # Validate Python script if provided for row or file scope
        if check_update.python_script and (
            check.scope == CheckScope.ROW.value or check.scope == CheckScope.FILE.value
        ):
            is_valid, error_msg = validate_python_script(check_update.python_script)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid Python script: {error_msg}",
                )
        check.python_script = check_update.python_script

    # Allow toggling the system flag if provided
    if hasattr(check_update, "is_system") and check_update.is_system is not None:
        check.is_system = check_update.is_system

    db.commit()
    db.refresh(check)

    return check


@router.delete("/{check_id}")
async def delete_check(
    check_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a check"""
    stmt = select(Check).where(Check.id == check_id)
    result = db.execute(stmt)
    check = result.scalar_one_or_none()

    if not check:
        raise HTTPException(status_code=404, detail="Check not found")

    # Prevent deleting system checks
    if check.is_system:
        raise HTTPException(status_code=403, detail="System checks cannot be deleted")

    db.delete(check)
    db.commit()

    return {"message": "Check deleted successfully"}


# Endpoint to validate Python script
@router.post("/validate-python")
async def validate_python(
    script: str,
    current_user: User = Depends(validate_jwt),
):
    """Validate a Python script for syntax errors"""
    is_valid, error_msg = validate_python_script(script)
    return {"is_valid": is_valid, "error": error_msg if not is_valid else None}


# Utility functions for column mapping and check application
def infer_column_data_type(values: List[Any]) -> str:
    """Infer the data type of a column based on its values"""
    # Filter out None and empty values
    filtered_values = [val for val in values if val is not None and val != ""]

    if not filtered_values:
        return DataType.TEXT.value  # Default to text if no values

    # Check if all values are boolean
    if all(
        isinstance(val, bool)
        or val in [0, 1, "0", "1", "true", "false", "True", "False"]
        for val in filtered_values
    ):
        return DataType.BOOLEAN.value

    # Check if all values can be converted to float
    try:
        [float(val) for val in filtered_values]
        return DataType.NUMERIC.value
    except (ValueError, TypeError):
        pass

    # Check if all values can be parsed as dates
    try:
        pd.to_datetime(filtered_values)
        return DataType.DATE.value
    except (ValueError, TypeError):
        pass

    # Check if it's categorical (limited number of unique values)
    unique_values = set(filtered_values)
    if len(unique_values) <= 10 or len(unique_values) <= 0.2 * len(filtered_values):
        return DataType.CATEGORICAL.value

    # Default to text
    return DataType.TEXT.value


def map_column_to_checks(
    column_name: str,
    values: List[Any],
    column_mappings: List[ColumnMapping],
    checks: List[Check],
) -> Dict[str, Any]:
    """Map a column to appropriate checks based on its name and data type"""
    # Infer data type
    data_type = infer_column_data_type(values)

    # Find best matching column mapping
    best_match = None
    best_score = 0

    for mapping in column_mappings:
        # Check exact match
        if mapping.column_name.lower() == column_name.lower():
            best_match = mapping
            break

        # Check synonyms
        if mapping.synonyms and any(
            syn.lower() == column_name.lower() for syn in mapping.synonyms
        ):
            best_match = mapping
            break

        # Fuzzy match on column name
        score = fuzz.ratio(mapping.column_name.lower(), column_name.lower())
        if score > best_score and score >= 80:  # 80% similarity threshold
            best_score = score
            best_match = mapping

        # Fuzzy match on synonyms
        if mapping.synonyms:
            for syn in mapping.synonyms:
                score = fuzz.ratio(syn.lower(), column_name.lower())
                if score > best_score and score >= 80:
                    best_score = score
                    best_match = mapping

    # If we found a mapping, use its data type
    if best_match:
        data_type = best_match.data_type

    # Find applicable checks
    applicable_checks = [check for check in checks if check.data_type == data_type]

    return {
        "column_name": column_name,
        "inferred_data_type": data_type,
        "mapped_column": best_match.column_name if best_match else None,
        "applicable_checks": [
            {
                "id": check.id,
                "name": check.name,
                "description": check.description,
                "parameters": check.parameters,
            }
            for check in applicable_checks
        ],
    }


@router.post("/analyze-dataset/{dataset_id}")
async def analyze_dataset(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Analyze a dataset and suggest column mappings and checks"""
    # Get the dataset
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get the data
    data = dataset.data or []

    if not data:
        return {"message": "No data available for analysis", "columns": []}

    # Get all column mappings for the user
    mappings_stmt = select(ColumnMapping).where(
        ColumnMapping.user_id == current_user["user_id"]
    )
    mappings_result = db.execute(mappings_stmt)
    column_mappings = mappings_result.scalars().all()

    # Get all checks
    checks_stmt = select(Check)
    checks_result = db.execute(checks_stmt)
    checks = checks_result.scalars().all()

    # Analyze each column
    columns = []
    for column_name in data[0].keys():
        # Extract values for this column
        values = [patient.get(column_name) for patient in data]

        # Map column to checks
        column_mapping = map_column_to_checks(
            column_name, values, column_mappings, checks
        )
        columns.append(column_mapping)

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "total_records": len(data),
        "columns": columns,
    }


@router.post("/apply-checks/{dataset_id}")
async def apply_checks(
    dataset_id: int,
    column_checks: Dict[str, List[int]],  # Map of column names to check IDs
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Apply specified checks to a dataset"""
    # Get the dataset
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.id == dataset_id,
        SyntheticDataset.user_id == current_user["user_id"],
    )

    result = db.execute(stmt)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get the data
    data = dataset.data or []

    if not data:
        return {"message": "No data available for analysis", "results": {}}

    # Get all checks
    checks_stmt = select(Check).where(
        Check.id.in_(
            [check_id for check_ids in column_checks.values() for check_id in check_ids]
        )
    )
    checks_result = db.execute(checks_stmt)
    checks = {check.id: check for check in checks_result.scalars().all()}

    # Apply checks to each column
    results = {}
    applied_checks = {}

    for column_name, check_ids in column_checks.items():
        # Extract values for this column
        values = [patient.get(column_name) for patient in data]

        # Apply each check
        column_results = {}
        column_applied_checks = []

        for check_id in check_ids:
            if check_id not in checks:
                continue

            check = checks[check_id]

            # Get the implementation function
            implementation = CHECK_IMPLEMENTATIONS.get(check.implementation)
            if not implementation:
                continue

            # Apply the check with parameters
            try:
                if check.parameters:
                    result = implementation(values, **check.parameters)
                else:
                    result = implementation(values)

                column_results[check.name] = result
                column_applied_checks.append(
                    {"id": check.id, "name": check.name, "parameters": check.parameters}
                )
            except Exception as e:
                column_results[check.name] = {"error": str(e)}

        results[column_name] = column_results
        applied_checks[column_name] = column_applied_checks

    # Update the dataset with the applied checks and results
    dataset.applied_checks = applied_checks
    dataset.check_results = results
    db.commit()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "applied_checks": applied_checks,
        "results": results,
    }
