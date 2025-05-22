#!/usr/bin/env python3
"""
Utility script to initialize system checks for an organization.

Usage:
    python initialize_checks.py --organization-id 1

The organization ID is required.
"""

import argparse
import sys
import os

# Add the parent directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func
from db import SessionLocal
from models import Check, DataType, Organization


def init_system_checks(organization_id):
    """Initialize system checks for an organization

    Args:
        organization_id: The organization ID to associate with system checks.
    """
    if organization_id is None:
        print("No organization ID provided, skipping system checks initialization")
        return False

    # Verify the organization exists
    db = SessionLocal()
    try:
        org = db.query(Organization).filter(Organization.id == organization_id).first()
        if not org:
            print(f"Error: Organization with ID {organization_id} not found")
            return False

        # Check if we already have system checks for this organization
        stmt = (
            select(func.count())
            .select_from(Check)
            .where(
                (Check.is_system == True) & (Check.organization_id == organization_id)  # noqa
            )
        )
        result = db.execute(stmt)
        count = result.scalar_one()

        if count > 0:
            print(
                f"System checks already exist for organization {organization_id} ({count} checks found)"
            )
            return True  # System checks already exist

        # Define system checks
        system_checks = [
            {
                "name": "Range Compliance",
                "description": "Check if values comply with a specified range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 0, "max_val": 100},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Missing Values",
                "description": "Check for missing values in a column",
                "data_type": DataType.NUMERIC.value,
                "parameters": {},
                "implementation": "check_missing_values",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Missing Values (Categorical)",
                "description": "Check for missing values in a categorical column",
                "data_type": DataType.CATEGORICAL.value,
                "parameters": {},
                "implementation": "check_missing_values",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Missing Values (Date)",
                "description": "Check for missing values in a date column",
                "data_type": DataType.DATE.value,
                "parameters": {},
                "implementation": "check_missing_values",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Missing Values (Text)",
                "description": "Check for missing values in a text column",
                "data_type": DataType.TEXT.value,
                "parameters": {},
                "implementation": "check_missing_values",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Missing Values (Boolean)",
                "description": "Check for missing values in a boolean column",
                "data_type": DataType.BOOLEAN.value,
                "parameters": {},
                "implementation": "check_missing_values",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Outliers",
                "description": "Check for outliers using IQR method",
                "data_type": DataType.NUMERIC.value,
                "parameters": {},
                "implementation": "check_outliers",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Categorical Distribution",
                "description": "Check distribution of categorical values",
                "data_type": DataType.CATEGORICAL.value,
                "parameters": {},
                "implementation": "check_categorical_distribution",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Numeric Statistics",
                "description": "Calculate basic statistics for numeric values",
                "data_type": DataType.NUMERIC.value,
                "parameters": {},
                "implementation": "check_numeric_statistics",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Date Range",
                "description": "Check range of dates",
                "data_type": DataType.DATE.value,
                "parameters": {},
                "implementation": "check_date_range",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Text Length",
                "description": "Check length statistics for text values",
                "data_type": DataType.TEXT.value,
                "parameters": {},
                "implementation": "check_text_length",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Boolean Distribution",
                "description": "Check distribution of boolean values",
                "data_type": DataType.BOOLEAN.value,
                "parameters": {},
                "implementation": "check_boolean_distribution",
                "is_system": True,
                "organization_id": organization_id,
            },
            # Blood pressure specific checks
            {
                "name": "Blood Pressure Range",
                "description": "Check if blood pressure values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 80, "max_val": 150},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Diastolic Range",
                "description": "Check if diastolic blood pressure values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 50, "max_val": 110},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            # Lab value specific checks
            {
                "name": "HbA1c Range",
                "description": "Check if HbA1c values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 5, "max_val": 7},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "LDL Range",
                "description": "Check if LDL values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 50, "max_val": 250},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "HDL Range",
                "description": "Check if HDL values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 20, "max_val": 80},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Triglycerides Range",
                "description": "Check if triglycerides values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 50, "max_val": 500},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Total Cholesterol Range",
                "description": "Check if total cholesterol values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 100, "max_val": 300},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
            {
                "name": "Creatinine Range",
                "description": "Check if creatinine values are within normal range",
                "data_type": DataType.NUMERIC.value,
                "parameters": {"min_val": 0.5, "max_val": 1.5},
                "implementation": "check_range_compliance",
                "is_system": True,
                "organization_id": organization_id,
            },
        ]

        # Validate implementations
        #for check_data in system_checks:
        #    if check_data["implementation"] not in CHECK_IMPLEMENTATIONS:
        #        print(
        #            f"Warning: Implementation '{check_data['implementation']}' not found for check '{check_data['name']}'"
        #        )

        # Create system checks
        for check_data in system_checks:
            check = Check(**check_data)
            db.add(check)

        db.commit()
        print(
            f"Created {len(system_checks)} system checks for organization {organization_id}"
        )
        return True

    except Exception as e:
        print(f"Error initializing system checks: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Initialize system checks for an organization"
    )
    parser.add_argument(
        "--organization-id",
        type=int,
        required=True,
        help="Organization ID to initialize checks for",
    )

    args = parser.parse_args()
    success = init_system_checks(args.organization_id)

    if success:
        print("System checks initialization completed successfully")
    else:
        print("System checks initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
