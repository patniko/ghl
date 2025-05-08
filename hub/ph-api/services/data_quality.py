import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db
from models import User, SyntheticDataset

router = APIRouter()


def calculate_statistics(values: List[float]) -> Dict[str, Any]:
    """Calculate statistical measures for a list of numerical values"""
    if not values or len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "outliers": None,
            "outlier_percentage": None,
            "q1": None,
            "q3": None,
        }

    # Basic statistics
    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    # Higher-order statistics
    skewness = None
    kurtosis = None

    if len(values) > 2 and std_dev > 0:
        # Calculate skewness
        skewness = np.sum(((values - mean) / std_dev) ** 3) / len(values)

        # Calculate kurtosis (excess kurtosis, normal = 0)
        if len(values) > 3:
            kurtosis = np.sum(((values - mean) / std_dev) ** 4) / len(values) - 3

    # Calculate quartiles and outliers using IQR method
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [val for val in values if val < lower_bound or val > upper_bound]
    outlier_percentage = (len(outliers) / len(values)) * 100

    return {
        "mean": float(mean),
        "median": float(median),
        "std_dev": float(std_dev),
        "min": float(min_val),
        "max": float(max_val),
        "skewness": float(skewness) if skewness is not None else None,
        "kurtosis": float(kurtosis) if kurtosis is not None else None,
        "outliers": len(outliers),
        "outlier_percentage": float(outlier_percentage),
        "q1": float(q1),
        "q3": float(q3),
    }


def calculate_distribution(values: List[float], bins: int = 10) -> Dict[str, Any]:
    """Calculate histogram data for distribution visualization"""
    if not values or len(values) == 0:
        return {"bins": [], "counts": []}

    # Calculate histogram
    counts, bin_edges = np.histogram(values, bins=bins)

    # Convert to list for JSON serialization
    bins_list = [float(bin_edge) for bin_edge in bin_edges[:-1]]
    counts_list = [int(count) for count in counts]

    return {"bins": bins_list, "counts": counts_list}


def analyze_categorical_variable(values: List[str]) -> Dict[str, Any]:
    """Analyze a categorical variable"""
    if not values or len(values) == 0:
        return {"value_counts": {}, "diversity_index": None}

    # Filter out None values
    filtered_values = [val for val in values if val is not None]

    if len(filtered_values) == 0:
        return {"value_counts": {}, "diversity_index": None}

    # Calculate value counts
    unique_values = set(filtered_values)
    value_counts = {val: filtered_values.count(val) for val in unique_values}

    # Calculate proportions
    total = len(filtered_values)
    proportions = {val: count / total for val, count in value_counts.items()}

    # Calculate Shannon's Diversity Index
    diversity_index = -sum(p * np.log(p) for p in proportions.values() if p > 0)

    # Add percentages to value counts
    value_counts_with_pct = {
        val: {"count": count, "percentage": proportions[val] * 100}
        for val, count in value_counts.items()
    }

    return {
        "value_counts": value_counts_with_pct,
        "diversity_index": float(diversity_index),
    }


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


@router.get("/{dataset_id}/quality-metrics")
async def get_quality_metrics(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get quality metrics for a specific synthetic dataset"""
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
        return {"message": "No data available for analysis", "metrics": {}}

    # Define key metrics to analyze
    key_metrics = [
        "age",
        "bp_systolic",
        "bp_diastolic",
        "pulse_entry",
        "resp_rate_entry",
        "spo2_entry",
        "height_entry",
        "weight_entry",
        "HbA1c",
        "Hb",
        "triglycerides_mg_dl",
        "tot_cholesterol_mg_dl",
        "HDL_mg_dl",
        "LDL_mg_dl",
        "VLDL_mg_dl",
        "creatinine_mg_dl",
    ]

    # Define categorical variables to analyze
    categorical_vars = [
        "sex",
        "verbal_consent",
        "cognit_impaired",
        "literate",
        "smoking_1",
        "smoking_2",
        "smoking_3",
    ]

    # Initialize results
    metrics = {
        "numerical": {},
        "categorical": {},
        "missing_values": {},
        "range_compliance": {},
    }

    # Define expected ranges for key metrics
    range_requirements = {
        "age": (30, 99),
        "bp_systolic": (80, 150),
        "bp_diastolic": (50, 110),
        "pulse_entry": (50, 100),
        "resp_rate_entry": (8, 20),
        "spo2_entry": (90, 100),
        "height_entry": (135, 190),
        "weight_entry": (40, 100),
        "HbA1c": (5, 7),
        "LDL_mg_dl": (50, 250),
        "HDL_mg_dl": (20, 80),
        "triglycerides_mg_dl": (50, 500),
        "tot_cholesterol_mg_dl": (100, 300),
        "creatinine_mg_dl": (0.5, 1.5),
    }

    # Analyze numerical variables
    for metric in key_metrics:
        # Extract values, handling missing data
        values = [patient.get(metric) for patient in data]
        values = [float(val) for val in values if val is not None and val != ""]

        if values:
            # Calculate statistics
            stats = calculate_statistics(values)

            # Calculate distribution
            distribution = calculate_distribution(values)

            # Store results
            metrics["numerical"][metric] = {
                "statistics": stats,
                "distribution": distribution,
            }

            # Check range compliance if range is defined
            if metric in range_requirements:
                min_val, max_val = range_requirements[metric]
                compliance = check_range_compliance(values, min_val, max_val)
                metrics["range_compliance"][metric] = {
                    "min": min_val,
                    "max": max_val,
                    **compliance,
                }

    # Analyze categorical variables
    for var in categorical_vars:
        # Extract values
        values = [patient.get(var) for patient in data]

        # Analyze
        analysis = analyze_categorical_variable(values)

        # Store results
        metrics["categorical"][var] = analysis

    # Calculate missing values for all fields
    total_records = len(data)
    for field in set(key_metrics + categorical_vars):
        missing_count = sum(
            1
            for patient in data
            if patient.get(field) is None or patient.get(field) == ""
        )
        missing_percentage = (
            (missing_count / total_records) * 100 if total_records > 0 else 0
        )

        metrics["missing_values"][field] = {
            "count": missing_count,
            "percentage": float(missing_percentage),
        }

    return {"total_records": total_records, "metrics": metrics}


@router.get("/dashboard-statistics")
async def get_dashboard_statistics(
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get aggregated statistics for the dashboard"""
    # Get all datasets for the current user
    stmt = select(SyntheticDataset).where(
        SyntheticDataset.user_id == current_user["user_id"]
    )

    result = db.execute(stmt)
    datasets = result.scalars().all()

    # Initialize statistics
    total_datasets = len(datasets)
    total_participants = 0
    active_participants = 0
    data_quality_scores = []
    completion_rates = []
    last_updated = None

    # Process each dataset
    for dataset in datasets:
        # Count participants
        patients = dataset.data or []
        total_participants += len(patients)

        # Estimate active participants (those with recent data entries)
        # For this example, we'll consider all participants as active
        active_participants += len(patients)

        # Calculate data quality score for this dataset
        if patients:
            # Check for missing values as a simple quality metric
            total_fields = 0
            missing_fields = 0

            for patient in patients:
                for field in patient:
                    total_fields += 1
                    if patient[field] is None or patient[field] == "":
                        missing_fields += 1

            # Quality score is the percentage of non-missing values
            if total_fields > 0:
                quality_score = 100 - (missing_fields / total_fields * 100)
                data_quality_scores.append(quality_score)

            # Calculate completion rate (simplified example)
            # Here we'll use a random value between 70-100% for demonstration
            # In a real app, this would be based on actual completion criteria
            completion_rate = np.random.uniform(70, 100)
            completion_rates.append(completion_rate)

        # Track the most recent update
        if last_updated is None or dataset.updated_at > last_updated:
            last_updated = dataset.updated_at

    # Calculate aggregate statistics
    avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0
    avg_completion_rate = np.mean(completion_rates) if completion_rates else 0

    # Format the response
    return {
        "totalDatasets": total_datasets,
        "totalParticipants": total_participants,
        "activeParticipants": active_participants,
        "dataQuality": round(avg_data_quality, 1),
        "completionRate": round(avg_completion_rate, 1),
        "lastUpdated": last_updated.isoformat() if last_updated else None,
    }


@router.get("/{dataset_id}/data-quality-checks")
async def get_data_quality_checks(
    dataset_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get data quality checks for a specific synthetic dataset"""
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
        return {"message": "No data available for analysis", "checks": []}

    # Define quality checks based on requirements
    quality_checks = [
        # General - Data Architecture
        {
            "category": "General - Data Architecture",
            "requirement": "Identifiers",
            "validation_question": "Is a unique alphanumeric identifier generated for each respondent?",
            "required_range": "Alphanumeric",
        },
        # Demographics
        {
            "category": "Demographics",
            "requirement": "Age",
            "validation_question": "Does the field accept only whole numbers?",
            "required_range": "30-99 years",
        },
        {
            "category": "Demographics",
            "requirement": "Biological Sex",
            "validation_question": "Are only allowed options available?",
            "required_range": "Female, Male, Other",
        },
        # Healthcare Visit
        {
            "category": "Healthcare Visit",
            "requirement": "Location",
            "validation_question": "Are all four locations available?",
            "required_range": "Urban-LCECU Hospital, Rural, CHAD Hospital, CMC Cardiology",
        },
        {
            "category": "Healthcare Visit",
            "requirement": "Setting",
            "validation_question": "Are only two options available?",
            "required_range": "Outpatient, Inpatient",
        },
        # Behavioral Data
        {
            "category": "Behavioral Data",
            "requirement": "Smoking Years",
            "validation_question": "Does it enforce valid range?",
            "required_range": "0-99 years",
        },
        {
            "category": "Behavioral Data",
            "requirement": "Daily Tobacco",
            "validation_question": "Does it enforce valid range?",
            "required_range": "0-50 units/day",
        },
        {
            "category": "Behavioral Data",
            "requirement": "Alcohol Consumption",
            "validation_question": "Does it enforce valid range?",
            "required_range": "0-50 drinks/week",
        },
        # Lab Results
        {
            "category": "Lab Results",
            "requirement": "LDL",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "50-250 mg/dL",
        },
        {
            "category": "Lab Results",
            "requirement": "HDL",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "20-80 mg/dL",
        },
        {
            "category": "Lab Results",
            "requirement": "Triglycerides",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "50-500 mg/dL",
        },
        {
            "category": "Lab Results",
            "requirement": "Total Cholesterol",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "100-300 mg/dL",
        },
        {
            "category": "Lab Results",
            "requirement": "HbA1C",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "5-7%",
        },
        {
            "category": "Lab Results",
            "requirement": "Creatinine",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "0.5-1.5 mg/dL",
        },
        # Biometrics
        {
            "category": "Biometrics",
            "requirement": "Blood Pressure (Systolic)",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "80-150 mmHg",
        },
        {
            "category": "Biometrics",
            "requirement": "Blood Pressure (Diastolic)",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "50-110 mmHg",
        },
        {
            "category": "Biometrics",
            "requirement": "Pulse Rate",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "50-100 per minute",
        },
        {
            "category": "Biometrics",
            "requirement": "SpO2",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "90-100%",
        },
        {
            "category": "Biometrics",
            "requirement": "Respiratory Rate",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "8-20 per minute",
        },
        {
            "category": "Biometrics",
            "requirement": "Weight",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "40-100 kg",
        },
        {
            "category": "Biometrics",
            "requirement": "Height",
            "validation_question": "Does it flag out-of-range values?",
            "required_range": "135-190 cm",
        },
    ]

    # Perform checks
    for check in quality_checks:
        # Initialize status as not applicable
        check["status"] = "not-applicable"
        check["actual_value"] = None
        check["notes"] = None

        # Check identifiers
        if check["requirement"] == "Identifiers":
            patient_ids = [patient.get("patient_ngsci_id") for patient in data]
            unique_ids = set(patient_ids)

            if len(unique_ids) == len(patient_ids):
                check["status"] = "pass"
                check["actual_value"] = f"{len(unique_ids)}/{len(patient_ids)} unique"
            else:
                check["status"] = "fail"
                check["actual_value"] = f"{len(unique_ids)}/{len(patient_ids)} unique"
                check["notes"] = "Duplicate patient IDs detected"

        # Check age
        elif check["requirement"] == "Age":
            ages = [
                patient.get("age") for patient in data if patient.get("age") is not None
            ]

            if ages:
                # Check if all ages are whole numbers
                all_whole = all(float(age).is_integer() for age in ages)

                # Check if ages are in range
                in_range = all(30 <= float(age) <= 99 for age in ages)

                if all_whole and in_range:
                    check["status"] = "pass"
                    check["actual_value"] = f"{min(ages)}-{max(ages)}"
                elif all_whole:
                    check["status"] = "warning"
                    check["actual_value"] = f"{min(ages)}-{max(ages)}"
                    check["notes"] = "Some ages are outside the expected range"
                else:
                    check["status"] = "fail"
                    check["actual_value"] = f"{min(ages)}-{max(ages)}"
                    check["notes"] = "Not all ages are whole numbers"

        # Check biological sex
        elif check["requirement"] == "Biological Sex":
            sexes = [
                patient.get("sex") for patient in data if patient.get("sex") is not None
            ]

            if sexes:
                allowed_options = ["Female", "Male", "Other"]
                invalid_values = [sex for sex in sexes if sex not in allowed_options]

                if not invalid_values:
                    check["status"] = "pass"
                    check["actual_value"] = "All values valid"
                else:
                    check["status"] = "fail"
                    check["actual_value"] = f"{len(invalid_values)} invalid values"
                    check["notes"] = (
                        f"Invalid values: {', '.join(set(invalid_values)[:3])}"
                    )
                    if len(set(invalid_values)) > 3:
                        check["notes"] += "..."

        # Check numerical ranges for lab results and biometrics
        elif check["category"] in ["Lab Results", "Biometrics", "Behavioral Data"]:
            field_mapping = {
                "LDL": "LDL_mg_dl",
                "HDL": "HDL_mg_dl",
                "Triglycerides": "triglycerides_mg_dl",
                "Total Cholesterol": "tot_cholesterol_mg_dl",
                "HbA1C": "HbA1c",
                "Creatinine": "creatinine_mg_dl",
                "Blood Pressure (Systolic)": "bp_systolic",
                "Blood Pressure (Diastolic)": "bp_diastolic",
                "Pulse Rate": "pulse_entry",
                "SpO2": "spo2_entry",
                "Respiratory Rate": "resp_rate_entry",
                "Weight": "weight_entry",
                "Height": "height_entry",
                "Smoking Years": "smoking_years",
                "Daily Tobacco": "daily_tobacco",
                "Alcohol Consumption": "alcohol_consumption",
            }

            field = field_mapping.get(check["requirement"])

            if field and any(field in patient for patient in data):
                values = [
                    float(patient.get(field))
                    for patient in data
                    if patient.get(field) is not None and patient.get(field) != ""
                ]

                if values:
                    # Parse required range
                    range_str = check["required_range"]
                    # Handle different range formats
                    if "-" in range_str:
                        # Format like "50-250 mg/dL" or "5-7%"
                        range_parts = range_str.split(" ")[0].split("-")

                        # Extract numeric part from min value
                        min_str = range_parts[0]
                        min_val = float(
                            "".join(c for c in min_str if c.isdigit() or c == ".")
                        )

                        # Extract numeric part from max value
                        max_str = range_parts[1]
                        max_val = float(
                            "".join(c for c in max_str if c.isdigit() or c == ".")
                        )
                    else:
                        # Default range if parsing fails
                        min_val = 0
                        max_val = 100

                    # Check if values are in range
                    out_of_range = [
                        val for val in values if val < min_val or val > max_val
                    ]

                    if not out_of_range:
                        check["status"] = "pass"
                        check["actual_value"] = f"{min(values)}-{max(values)}"
                    elif (
                        len(out_of_range) < len(values) * 0.05
                    ):  # Less than 5% out of range
                        check["status"] = "warning"
                        check["actual_value"] = f"{min(values)}-{max(values)}"
                        check["notes"] = f"{len(out_of_range)} values outside range"
                    else:
                        check["status"] = "fail"
                        check["actual_value"] = f"{min(values)}-{max(values)}"
                        check["notes"] = f"{len(out_of_range)} values outside range"
            else:
                check["notes"] = f"Field {field} not present in dataset"

    return {"total_records": len(data), "checks": quality_checks}
