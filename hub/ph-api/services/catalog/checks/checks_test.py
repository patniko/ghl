from pytest import Session
from sqlalchemy import text

from db import get_db
from models import (
    User,
    Check,
    ColumnMapping,
    SyntheticDataset,
    DataType,
    Organization,
    UserOrganization,
)
from auth import create_access_token

from tests.test_factory import create_test_client

client = create_test_client()


# Helper functions for check catalog tests
def setup_test_user(db: Session):
    """Create a test user for check catalog tests"""
    # Generate unique email and phone to avoid conflicts
    import uuid

    unique_id = str(uuid.uuid4())[:8]

    # First create a test organization
    org = db.query(Organization).filter(Organization.slug == "test-org").first()
    if not org:
        org = Organization(
            name="Test Organization",
            slug="test-org",
            description="Test organization for automated tests",
        )
        db.add(org)
        db.flush()

    # First clean up any existing test users to avoid conflicts
    try:
        users = db.query(User).filter(User.email.like("test-%@example.com")).all()

        for user in users:
            # Use direct SQL to avoid cascade loading issues
            db.execute(text(f"DELETE FROM users WHERE id = {user.id}"))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error cleaning up test users: {str(e)}")

    # Create new user with unique identifiers
    user = User(
        first_name="Test",
        last_name="User",
        email=f"test-{unique_id}@example.com",
        email_verified=False,
        picture="",
    )
    db.add(user)
    db.flush()  # Flush to get the user ID

    # Create user-organization relationship
    user_org = UserOrganization(user_id=user.id, organization_id=org.id, is_admin=True)
    db.add(user_org)
    db.commit()
    db.refresh(user)
    return user


def teardown_test_user(db: Session, user_id: int):
    """Clean up test user and associated data"""
    try:
        # First rollback any pending transactions
        db.rollback()

        # Delete column mappings with direct SQL
        db.execute(text(f"DELETE FROM column_mappings WHERE user_id = {user_id}"))
        db.flush()

        # Delete user-created checks (non-system checks)
        checks = db.query(Check).filter(Check.is_system == False).all()  # noqa
        for check in checks:
            db.delete(check)
        db.flush()

        # Delete synthetic datasets with direct SQL
        db.execute(text(f"DELETE FROM synthetic_datasets WHERE user_id = {user_id}"))
        db.flush()

        # Delete user-organization relationships
        db.execute(text(f"DELETE FROM user_organizations WHERE user_id = {user_id}"))
        db.flush()

        # Finally delete the user with direct SQL
        if user_id:
            db.execute(text(f"DELETE FROM users WHERE id = {user_id}"))
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in teardown: {str(e)}")


def create_test_check(
    db: Session, name: str, data_type: str, implementation: str, is_system: bool = False
):
    """Create a test check"""
    # Get the test organization
    org = db.query(Organization).filter(Organization.slug == "test-org").first()

    check = Check(
        organization_id=org.id,
        name=name,
        description=f"Test {name} description",
        data_type=data_type,
        parameters={},
        implementation=implementation,
        is_system=is_system,
        python_script="",  # Add empty python_script field
        scope="field",  # Add scope field
    )
    db.add(check)
    db.commit()
    db.refresh(check)
    return check


def create_test_column_mapping(
    db: Session, user_id: int, column_name: str, data_type: str
):
    """Create a test column mapping"""
    # Get the test organization
    org = db.query(Organization).filter(Organization.slug == "test-org").first()

    mapping = ColumnMapping(
        organization_id=org.id,
        user_id=user_id,
        column_name=column_name,
        data_type=data_type,
        description=f"Test {column_name} description",
        synonyms=[f"{column_name}_alt", f"{column_name}_synonym"],
    )
    db.add(mapping)
    db.commit()
    db.refresh(mapping)
    return mapping


def create_test_dataset(db: Session, user_id: int, name: str = "Test Dataset"):
    """Create a test synthetic dataset"""
    # Get the test organization
    org = db.query(Organization).filter(Organization.slug == "test-org").first()

    # Create sample patient data
    patients = [
        {
            "patient_id": "P001",
            "age": 45,
            "sex": "M",
            "height": 175.5,
            "weight": 80.2,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "smoking": "No",
        },
        {
            "patient_id": "P002",
            "age": 62,
            "sex": "F",
            "height": 160.0,
            "weight": 65.5,
            "bp_systolic": 135,
            "bp_diastolic": 85,
            "smoking": "Yes",
        },
        {
            "patient_id": "P003",
            "age": 28,
            "sex": "M",
            "height": 182.3,
            "weight": 78.1,
            "bp_systolic": 118,
            "bp_diastolic": 75,
            "smoking": "No",
        },
    ]

    dataset = SyntheticDataset(
        organization_id=org.id,
        user_id=user_id,
        name=name,
        description="Test dataset description",
        num_patients=len(patients),
        data=patients,
        column_mappings={},
        applied_checks={},
        check_results={},
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_auth_token(user_id: int, email: str = None):
    """Create an authentication token for the test user"""
    # Get the test organization
    db = next(get_db())
    _ = db.query(Organization).filter(Organization.slug == "test-org").first()

    if email is None:
        email = f"test-{user_id}@example.com"

    access_token = create_access_token(
        data={
            "sub": str(user_id),
            "user_id": user_id,
            "first_name": "Test",
            "last_name": "User",
            "email": email,
            "email_verified": False,
            "picture": "",
            "updated_at": "2023-01-01T00:00:00Z",
            "admin": False,
            "password_set": False,
            "is_admin": True,
        }
    )
    return access_token


# Test the check implementation functions directly
def test_check_range_compliance():
    """Test the range compliance check function"""
    from services.catalog.checks.catalog import check_range_compliance

    # Test with values in range
    values = [5, 10, 15, 20, 25]
    result = check_range_compliance(values, 0, 30)
    assert result["in_range_count"] == 5
    assert result["out_of_range_count"] == 0
    assert result["compliance_percentage"] == 100.0

    # Test with some values out of range
    values = [5, 10, 35, 40, 25]
    result = check_range_compliance(values, 0, 30)
    assert result["in_range_count"] == 3
    assert result["out_of_range_count"] == 2
    assert result["compliance_percentage"] == 60.0

    # Test with empty list
    result = check_range_compliance([], 0, 30)
    assert result["in_range_count"] == 0
    assert result["out_of_range_count"] == 0
    assert result["compliance_percentage"] == 0


def test_check_missing_values():
    """Test the missing values check function"""
    from services.catalog.checks.catalog import check_missing_values

    # Test with no missing values
    values = [1, 2, 3, 4, 5]
    result = check_missing_values(values)
    assert result["total_count"] == 5
    assert result["missing_count"] == 0
    assert result["missing_percentage"] == 0.0

    # Test with some missing values
    values = [1, None, 3, "", float("nan")]
    result = check_missing_values(values)
    assert result["total_count"] == 5
    assert result["missing_count"] == 3
    assert result["missing_percentage"] == 60.0

    # Test with empty list
    result = check_missing_values([])
    assert result["total_count"] == 0
    assert result["missing_count"] == 0
    assert result["missing_percentage"] == 100.0


def test_check_outliers():
    """Test the outliers check function"""
    from services.catalog.checks.catalog import check_outliers

    # Test with no outliers
    values = [10, 12, 11, 13, 14, 12, 11]
    result = check_outliers(values)
    assert result["total_count"] == 7
    assert result["outlier_count"] == 0
    assert result["outlier_percentage"] == 0.0
    assert len(result["outliers"]) == 0

    # Test with outliers
    values = [10, 12, 11, 13, 14, 50, 100]
    result = check_outliers(values)
    assert result["total_count"] == 7
    assert result["outlier_count"] == 1
    assert result["outlier_percentage"] > 0.0
    assert len(result["outliers"]) == 1
    assert 100 in result["outliers"]

    # Test with too few values
    result = check_outliers([1, 2])
    assert result["total_count"] == 2
    assert result["outlier_count"] == 0
    assert result["outlier_percentage"] == 0.0


def test_check_categorical_distribution():
    """Test the categorical distribution check function"""
    from services.catalog.checks.catalog import check_categorical_distribution

    # Test with categorical values
    values = ["A", "B", "A", "C", "B", "A", "A"]
    result = check_categorical_distribution(values)
    assert result["total_count"] == 7
    assert result["unique_count"] == 3
    assert "A" in result["distribution"]
    assert "B" in result["distribution"]
    assert "C" in result["distribution"]
    assert result["distribution"]["A"]["count"] == 4
    assert result["distribution"]["B"]["count"] == 2
    assert result["distribution"]["C"]["count"] == 1

    # Test with empty list
    result = check_categorical_distribution([])
    assert result["total_count"] == 0
    assert result["unique_count"] == 0
    assert result["distribution"] == {}


def test_check_numeric_statistics():
    """Test the numeric statistics check function"""
    from services.catalog.checks.catalog import check_numeric_statistics

    # Test with numeric values
    values = [10, 20, 30, 40, 50]
    result = check_numeric_statistics(values)
    assert result["count"] == 5
    assert result["mean"] == 30.0
    assert result["median"] == 30.0
    assert result["min"] == 10.0
    assert result["max"] == 50.0
    assert result["std_dev"] > 0.0

    # Test with empty list
    result = check_numeric_statistics([])
    assert result["count"] == 0
    assert result["mean"] is None
    assert result["median"] is None
    assert result["min"] is None
    assert result["max"] is None
    assert result["std_dev"] is None


def test_infer_column_data_type():
    """Test the column data type inference function"""
    from services.catalog.checks.catalog import infer_column_data_type

    # Test numeric data
    assert infer_column_data_type([1, 2, 3, 4, 5]) == DataType.NUMERIC.value
    assert infer_column_data_type(["1", "2.5", "3", "4.2"]) == DataType.NUMERIC.value

    # Test boolean data
    assert infer_column_data_type([True, False, True]) == DataType.BOOLEAN.value
    assert infer_column_data_type([0, 1, 0, 1]) == DataType.BOOLEAN.value
    assert infer_column_data_type(["true", "false", "True"]) == DataType.BOOLEAN.value

    # Test date data
    assert infer_column_data_type(["2023-01-01", "2023-02-15"]) == DataType.DATE.value

    # Test categorical data (few unique values)
    assert (
        infer_column_data_type(["A", "B", "A", "C", "B", "A"])
        == DataType.CATEGORICAL.value
    )

    # Test text data (many unique values)
    assert (
        infer_column_data_type(
            [
                "text1",
                "text2",
                "text3",
                "text4",
                "text5",
                "text6",
                "text7",
                "text8",
                "text9",
                "text10",
                "text11",
            ]
        )
        == DataType.TEXT.value
    )

    # Test empty list
    assert infer_column_data_type([]) == DataType.TEXT.value


# API Tests
def test_create_check():
    """Test creating a new check"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create check
        check_data = {
            "name": "Test Range Check",
            "description": "Check if values are within a specified range",
            "data_type": DataType.NUMERIC.value,
            "parameters": {"min_val": 0, "max_val": 100},
            "implementation": "check_range_compliance",
            "scope": "field",
            "python_script": "",
        }

        # Get the test organization
        org = db.query(Organization).filter(Organization.slug == "test-org").first()

        response = client.post(
            f"/{org.slug}/catalog/checks",
            json=check_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == check_data["name"]
        assert data["description"] == check_data["description"]
        assert data["data_type"] == check_data["data_type"]
        assert data["parameters"] == check_data["parameters"]
        assert data["implementation"] == check_data["implementation"]
        assert data["is_system"] is False

    finally:
        teardown_test_user(db, user.id)


def test_create_check_invalid_implementation():
    """Test creating a check with an invalid implementation"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create check with invalid implementation
        check_data = {
            "name": "Invalid Check",
            "description": "This check has an invalid implementation",
            "data_type": DataType.NUMERIC.value,
            "parameters": {},
            "implementation": "nonexistent_implementation",
        }
        response = client.post(
            "/test-org/catalog/checks",
            json=check_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400
        assert (
            "Implementation 'nonexistent_implementation' not found"
            in response.json()["detail"]
        )

    finally:
        teardown_test_user(db, user.id)


def test_get_checks():
    """Test retrieving all checks"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create multiple checks
        check1 = create_test_check(
            db, "Numeric Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )
        check2 = create_test_check(
            db,
            "Categorical Check",
            DataType.CATEGORICAL.value,
            "check_categorical_distribution",
        )

        # Get all checks
        response = client.get(
            "/test-org/catalog/checks", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2  # There might be system checks already
        assert any(c["id"] == check1.id for c in data)
        assert any(c["id"] == check2.id for c in data)

    finally:
        teardown_test_user(db, user.id)


def test_get_checks_with_filters():
    """Test retrieving checks with filters"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create checks of different types
        check1 = create_test_check(
            db, "Numeric Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )
        check2 = create_test_check(
            db,
            "Categorical Check",
            DataType.CATEGORICAL.value,
            "check_categorical_distribution",
        )
        check3 = create_test_check(
            db, "System Check", DataType.NUMERIC.value, "check_outliers", is_system=True
        )

        # Get checks with numeric data type
        response = client.get(
            f"/test-org/catalog/checks?data_type={DataType.NUMERIC.value}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert any(c["id"] == check1.id for c in data)
        assert not any(c["id"] == check2.id for c in data)
        assert any(c["id"] == check3.id for c in data)

        # Get system checks
        response = client.get(
            "/test-org/catalog/checks?is_system=true",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = response.json()

        # Try to delete system check (should be forbidden)
        response = client.delete(
            f"/test-org/catalog/checks/{check3.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "System checks cannot be deleted"

        # Verify system checks response
        assert not any(c["id"] == check1.id for c in data)
        assert not any(c["id"] == check2.id for c in data)
        assert any(c["id"] == check3.id for c in data)

    finally:
        teardown_test_user(db, user.id)


def test_get_check():
    """Test retrieving a specific check"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create check
        check = create_test_check(
            db, "Test Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )

        # Get check
        response = client.get(
            f"/test-org/catalog/checks/{check.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == check.id
        assert data["name"] == check.name
        assert data["data_type"] == check.data_type
        assert data["implementation"] == check.implementation

    finally:
        teardown_test_user(db, user.id)


def test_get_nonexistent_check():
    """Test retrieving a check that doesn't exist"""
    db = next(get_db())
    user = None
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Get non-existent check
        response = client.get(
            "/test-org/catalog/checks/9999",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Check not found"

    finally:
        if user:
            teardown_test_user(db, user.id)


def test_update_check():
    """Test updating a check"""
    db = next(get_db())
    user = None
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create check
        check = create_test_check(
            db, "Test Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )

        # Update check
        update_data = {
            "name": "Updated Check Name",
            "description": "Updated check description",
            "parameters": {"new_param": "value"},
            "python_script": "",
        }
        response = client.put(
            f"/test-org/catalog/checks/{check.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == check.id
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["parameters"] == update_data["parameters"]
        assert data["implementation"] == check.implementation  # Unchanged

    finally:
        teardown_test_user(db, user.id)


def test_update_system_check():
    """Test updating a system check (should be forbidden)"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create system check
        check = create_test_check(
            db,
            "System Check",
            DataType.NUMERIC.value,
            "check_numeric_statistics",
            is_system=True,
        )

        # Try to update system check
        update_data = {
            "name": "Updated System Check",
            "description": "This update should fail",
            "python_script": "",
        }
        response = client.put(
            f"/test-org/catalog/checks/{check.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Try to delete system check
        response = client.delete(
            f"/test-org/catalog/checks/{check.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "System checks cannot be deleted"

    finally:
        teardown_test_user(db, user.id)


def test_delete_check():
    """Test deleting a check"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create check
        check = create_test_check(
            db, "Test Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )

        # Delete check
        response = client.delete(
            f"/test-org/catalog/checks/{check.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Check deleted successfully"

        # Verify check is deleted
        response = client.get(
            f"/test-org/catalog/checks/{check.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    finally:
        teardown_test_user(db, user.id)


def test_delete_system_check():
    """Test deleting a system check (should be forbidden)"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create system check
        check = create_test_check(
            db,
            "System Check",
            DataType.NUMERIC.value,
            "check_numeric_statistics",
            is_system=True,
        )

        # Try to delete system check
        response = client.delete(
            f"/test-org/catalog/checks/{check.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "System checks cannot be deleted"

    finally:
        teardown_test_user(db, user.id)


def test_create_column_mapping():
    """Test creating a new column mapping"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create column mapping
        mapping_data = {
            "column_name": "age",
            "data_type": DataType.NUMERIC.value,
            "description": "Patient age in years",
            "synonyms": ["patient_age", "age_years"],
        }

        # Get the test organization
        org = db.query(Organization).filter(Organization.slug == "test-org").first()

        response = client.post(
            f"/{org.slug}/catalog/mappings",
            json=mapping_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["column_name"] == mapping_data["column_name"]
        assert data["data_type"] == mapping_data["data_type"]
        assert data["description"] == mapping_data["description"]
        assert data["synonyms"] == mapping_data["synonyms"]

    finally:
        teardown_test_user(db, user.id)


def test_get_column_mappings():
    """Test retrieving all column mappings for a user"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create multiple column mappings
        mapping1 = create_test_column_mapping(
            db, user.id, "age", DataType.NUMERIC.value
        )
        mapping2 = create_test_column_mapping(
            db, user.id, "sex", DataType.CATEGORICAL.value
        )

        # Get all mappings
        response = client.get(
            "/test-org/catalog/mappings", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert any(m["id"] == mapping1.id for m in data)
        assert any(m["id"] == mapping2.id for m in data)

    finally:
        teardown_test_user(db, user.id)


def test_get_column_mappings_with_filter():
    """Test retrieving column mappings with data type filter"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create mappings of different types
        mapping1 = create_test_column_mapping(
            db, user.id, "age", DataType.NUMERIC.value
        )
        _ = create_test_column_mapping(db, user.id, "sex", DataType.CATEGORICAL.value)

        # Get mappings with numeric data type
        response = client.get(
            f"/test-org/catalog/mappings?data_type={DataType.NUMERIC.value}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == mapping1.id
        assert data[0]["data_type"] == DataType.NUMERIC.value

    finally:
        teardown_test_user(db, user.id)


def test_get_column_mapping():
    """Test retrieving a specific column mapping"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create column mapping
        mapping = create_test_column_mapping(db, user.id, "age", DataType.NUMERIC.value)

        # Get mapping
        response = client.get(
            f"/test-org/catalog/mappings/{mapping.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mapping.id
        assert data["column_name"] == mapping.column_name
        assert data["data_type"] == mapping.data_type
        assert data["description"] == mapping.description
        assert data["synonyms"] == mapping.synonyms

    finally:
        teardown_test_user(db, user.id)


def test_update_column_mapping():
    """Test updating a column mapping"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create column mapping
        mapping = create_test_column_mapping(db, user.id, "age", DataType.NUMERIC.value)

        # Update mapping
        update_data = {
            "data_type": DataType.CATEGORICAL.value,
            "description": "Updated description",
            "synonyms": ["new_synonym1", "new_synonym2"],
        }
        response = client.put(
            f"/test-org/catalog/mappings/{mapping.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mapping.id
        assert data["column_name"] == mapping.column_name  # Unchanged
        assert data["data_type"] == update_data["data_type"]
        assert data["description"] == update_data["description"]
        assert data["synonyms"] == update_data["synonyms"]

    finally:
        teardown_test_user(db, user.id)


def test_delete_column_mapping():
    """Test deleting a column mapping"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create column mapping
        mapping = create_test_column_mapping(db, user.id, "age", DataType.NUMERIC.value)

        # Delete mapping
        response = client.delete(
            f"/test-org/catalog/mappings/{mapping.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Column mapping deleted successfully"

        # Verify mapping is deleted
        response = client.get(
            f"/test-org/catalog/mappings/{mapping.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    finally:
        teardown_test_user(db, user.id)


def test_analyze_dataset():
    """Test analyzing a dataset for column mappings and checks"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create dataset
        dataset = create_test_dataset(db, user.id)

        # Create column mappings
        _ = create_test_column_mapping(db, user.id, "age", DataType.NUMERIC.value)
        _ = create_test_column_mapping(db, user.id, "sex", DataType.CATEGORICAL.value)

        # Create checks
        numeric_check = create_test_check(
            db, "Numeric Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )
        categorical_check = create_test_check(
            db,
            "Categorical Check",
            DataType.CATEGORICAL.value,
            "check_categorical_distribution",
        )

        # Analyze dataset
        response = client.post(
            f"/test-org/catalog/checks/analyze-dataset/{dataset.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset.id
        assert data["dataset_name"] == dataset.name
        assert data["total_records"] == 3  # 3 test patients

        # Check that columns were analyzed
        assert len(data["columns"]) > 0

        # Check that age column was mapped correctly
        age_column = next(
            (c for c in data["columns"] if c["column_name"] == "age"), None
        )
        assert age_column is not None
        assert age_column["inferred_data_type"] == DataType.NUMERIC.value
        assert age_column["mapped_column"] == "age"
        assert len(age_column["applicable_checks"]) > 0
        assert any(c["id"] == numeric_check.id for c in age_column["applicable_checks"])

        # Check that sex column was mapped correctly
        sex_column = next(
            (c for c in data["columns"] if c["column_name"] == "sex"), None
        )
        assert sex_column is not None
        assert sex_column["inferred_data_type"] == DataType.CATEGORICAL.value
        assert sex_column["mapped_column"] == "sex"
        assert len(sex_column["applicable_checks"]) > 0
        assert any(
            c["id"] == categorical_check.id for c in sex_column["applicable_checks"]
        )

    finally:
        teardown_test_user(db, user.id)


def test_apply_checks():
    """Test applying checks to a dataset"""
    db = next(get_db())
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Create dataset
        dataset = create_test_dataset(db, user.id)

        # Create checks
        numeric_check = create_test_check(
            db, "Numeric Check", DataType.NUMERIC.value, "check_numeric_statistics"
        )
        categorical_check = create_test_check(
            db,
            "Categorical Check",
            DataType.CATEGORICAL.value,
            "check_categorical_distribution",
        )
        range_check = create_test_check(
            db,
            "Range Check",
            DataType.NUMERIC.value,
            "check_range_compliance",
            is_system=False,
        )

        # Set parameters for range check
        range_check.parameters = {"min_val": 0, "max_val": 100}
        db.commit()

        # Apply checks to dataset
        column_checks = {
            "age": [numeric_check.id, range_check.id],
            "sex": [categorical_check.id],
        }

        response = client.post(
            f"/test-org/catalog/checks/apply-checks/{dataset.id}",
            json=column_checks,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset.id
        assert data["dataset_name"] == dataset.name

        # Check that results were generated
        assert "results" in data
        assert "age" in data["results"]
        assert "sex" in data["results"]

        # Check numeric statistics results
        assert numeric_check.name in data["results"]["age"]
        age_stats = data["results"]["age"][numeric_check.name]
        assert age_stats["count"] == 3
        assert "mean" in age_stats
        assert "median" in age_stats
        assert "min" in age_stats
        assert "max" in age_stats

        # Check range compliance results
        assert range_check.name in data["results"]["age"]
        range_results = data["results"]["age"][range_check.name]
        assert range_results["in_range_count"] == 3  # All ages are between 0-100
        assert range_results["out_of_range_count"] == 0

        # Check categorical distribution results
        assert categorical_check.name in data["results"]["sex"]
        sex_dist = data["results"]["sex"][categorical_check.name]
        assert sex_dist["unique_count"] == 2  # M and F
        assert "M" in sex_dist["distribution"]
        assert "F" in sex_dist["distribution"]

        # Verify that the dataset was updated with applied checks and results
        db.refresh(dataset)
        assert dataset.applied_checks is not None
        assert dataset.check_results is not None
        assert "age" in dataset.applied_checks
        assert "sex" in dataset.applied_checks

    finally:
        teardown_test_user(db, user.id)


def test_apply_checks_nonexistent_dataset():
    """Test applying checks to a non-existent dataset"""
    db = next(get_db())
    user = None
    try:
        # Setup test user
        user = setup_test_user(db)
        token = get_auth_token(user.id)

        # Apply checks to non-existent dataset
        column_checks = {"age": [1, 2], "sex": [3]}
        response = client.post(
            "/test-org/catalog/checks/apply-checks/9999",
            json=column_checks,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Dataset not found"

    finally:
        if user:
            teardown_test_user(db, user.id)
