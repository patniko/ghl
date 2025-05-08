import uuid
import sqlalchemy.orm
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db
from models import Organization, User, Batch, File, DicomFile, SyntheticDataset, Model
from auth import create_access_token


def setup_test_organization(db: Session, slug: str = None):
    """
    Set up a test organization and admin user for testing.

    Args:
        db: Database session
        slug: Optional slug for the organization. If not provided, a unique slug will be generated.

    Returns:
        Tuple of (organization, admin_user)
    """
    # Generate a unique slug if not provided
    if not slug:
        unique_id = str(uuid.uuid4())[:8]
        slug = f"test-org-{unique_id}"

    try:
        # Check if organization already exists
        existing_org = db.query(Organization).filter(Organization.slug == slug).first()
        if existing_org:
            # Delete all related records first to avoid foreign key constraint violations
            db.execute(
                text(f"DELETE FROM models WHERE organization_id = {existing_org.id}")
            )
            db.execute(
                text(f"DELETE FROM checks WHERE organization_id = {existing_org.id}")
            )
            db.execute(
                text(
                    f"DELETE FROM column_mappings WHERE organization_id = {existing_org.id}"
                )
            )
            db.execute(
                text(f"DELETE FROM files WHERE organization_id = {existing_org.id}")
            )
            db.execute(
                text(
                    f"DELETE FROM dicom_files WHERE organization_id = {existing_org.id}"
                )
            )
            db.execute(
                text(
                    f"DELETE FROM synthetic_datasets WHERE organization_id = {existing_org.id}"
                )
            )
            db.execute(
                text(f"DELETE FROM users WHERE organization_id = {existing_org.id}")
            )
            db.execute(text(f"DELETE FROM organizations WHERE id = {existing_org.id}"))
            db.commit()

        # Create a new test organization
        org = Organization(
            name="Test Organization",
            slug=slug,
            description="Test organization for automated tests",
        )
        db.add(org)
        db.flush()

        # Get the ID but detach the object to avoid relationship loading
        org_id = org.id
        db.expunge(org)

        # Create admin user for this organization with unique email and phone
        _ = slug[-4:] if len(slug) >= 4 else slug
        admin = User(
            first_name="Test",
            last_name="Admin",
            email=f"admin-{slug}@example.com",
            email_verified=True,
            password_hash="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
            is_admin=True,
        )
        db.add(admin)
        db.flush()  # Flush to get the user ID

        # Create user-organization relationship
        from models import UserOrganization

        user_org = UserOrganization(
            user_id=admin.id, organization_id=org_id, is_admin=True
        )
        db.add(user_org)
        db.commit()

        # Reload the organization without relationships
        org = (
            db.query(Organization)
            .options(sqlalchemy.orm.noload("*"))
            .filter(Organization.id == org_id)
            .first()
        )

        db.refresh(admin)

        return org, admin
    except Exception as e:
        db.rollback()
        print(f"Error in setup_test_organization: {str(e)}")
        raise


def teardown_test_organization(db: Session, slug: str):
    """
    Clean up test organization and all related data after tests.

    Args:
        db: Database session
        slug: Organization slug to clean up
    """
    try:
        # Use direct SQL to avoid cascade loading issues
        org = db.query(Organization).filter(Organization.slug == slug).first()
        if org:
            db.execute(text(f"DELETE FROM models WHERE organization_id = {org.id}"))
            db.execute(text(f"DELETE FROM checks WHERE organization_id = {org.id}"))
            db.execute(
                text(f"DELETE FROM column_mappings WHERE organization_id = {org.id}")
            )
            db.execute(text(f"DELETE FROM files WHERE organization_id = {org.id}"))
            db.execute(
                text(f"DELETE FROM dicom_files WHERE organization_id = {org.id}")
            )
            db.execute(
                text(f"DELETE FROM synthetic_datasets WHERE organization_id = {org.id}")
            )
            # Delete user-organization relationships
            db.execute(
                text(f"DELETE FROM user_organizations WHERE organization_id = {org.id}")
            )
            # Find users associated with this organization
            user_ids = db.execute(
                text(
                    f"SELECT user_id FROM user_organizations WHERE organization_id = {org.id}"
                )
            ).fetchall()
            for user_id in user_ids:
                db.execute(text(f"DELETE FROM users WHERE id = {user_id[0]}"))
            db.commit()
            db.execute(text(f"DELETE FROM organizations WHERE id = {org.id}"))
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in teardown_test_organization: {str(e)}")


def get_auth_token(user_id: int, org_id: int = None):
    """
    Create an authentication token for the test user.

    Args:
        user_id: User ID
        org_id: Organization ID (optional, included for backward compatibility)

    Returns:
        JWT access token
    """
    access_token = create_access_token(
        data={
            "sub": str(user_id),
            "user_id": user_id,
            "first_name": "Test",
            "last_name": "User",
            "email": "test@example.com",
            "email_verified": False,
            "picture": "",
            "updated_at": "2023-01-01T00:00:00Z",
            "admin": False,
            "password_set": False,
            "is_admin": True,
        }
    )
    return access_token


def create_test_batch(db: Session, user_id: int, org_id: int, name: str = "Test Batch"):
    """
    Create a test batch for the specified user and organization.

    Args:
        db: Database session
        user_id: User ID
        org_id: Organization ID
        name: Batch name

    Returns:
        Created Batch object
    """
    batch = Batch(
        organization_id=org_id,
        user_id=user_id,
        name=name,
        description="Test batch description",
        quality_summary={
            "total_datasets": 0,
            "total_checks": 0,
            "checks_by_type": {},
            "issues_by_severity": {"info": 0, "warning": 0, "error": 0},
        },
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_test_file(
    db: Session, user_id: int, org_id: int, batch_id: int, file_type: str = "csv"
):
    """
    Create a test file for the specified batch.

    Args:
        db: Database session
        user_id: User ID
        org_id: Organization ID
        batch_id: Batch ID
        file_type: File type (csv, mp4, npz, etc.)

    Returns:
        Created File object
    """
    file = File(
        organization_id=org_id,
        user_id=user_id,
        batch_id=batch_id,
        filename=f"test.{file_type}",
        original_filename=f"test.{file_type}",
        file_path=f"/uploads/projects/00/batches/{batch_id}/test.{file_type}",
        file_size=1024,
        content_type="text/csv" if file_type == "csv" else f"application/{file_type}",
        file_type=file_type,
        file_metadata={},
        csv_headers=["col1", "col2", "col3"] if file_type == "csv" else None,
        processing_status="completed",
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def create_test_model(
    db: Session,
    org_id: int,
    name: str,
    implementation: str,
    version: str = "1.0",
    is_system: bool = False,
):
    """
    Create a test model.

    Args:
        db: Database session
        org_id: Organization ID
        name: Model name
        implementation: Model implementation
        version: Model version
        is_system: Whether this is a system model

    Returns:
        Created Model object
    """
    model = Model(
        organization_id=org_id,
        name=name,
        description=f"Test {name} description",
        version=version,
        parameters={},
        implementation=implementation,
        is_system=is_system,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def create_test_synthetic_dataset(
    db: Session, user_id: int, org_id: int, batch_id: int
):
    """
    Create a test synthetic dataset for the specified batch.

    Args:
        db: Database session
        user_id: User ID
        org_id: Organization ID
        batch_id: Batch ID

    Returns:
        Created SyntheticDataset object
    """
    dataset = SyntheticDataset(
        organization_id=org_id,
        user_id=user_id,
        batch_id=batch_id,
        name="Test Dataset",
        description="Test dataset description",
        num_patients=100,
        data={"patients": []},
        column_mappings={},
        applied_checks={},
        check_results={},
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def create_test_dicom_file(
    db: Session, user_id: int, org_id: int, batch_instance_uid: str
):
    """
    Create a test DICOM file.

    Args:
        db: Database session
        user_id: User ID
        org_id: Organization ID
        batch_instance_uid: DICOM batch instance UID

    Returns:
        Created DicomFile object
    """
    from datetime import datetime

    dicom_file = DicomFile(
        organization_id=org_id,
        user_id=user_id,
        filename="test.dcm",
        original_filename="test.dcm",
        file_path="//projects/test/batches/default/test.dcm",
        file_size=1024,
        content_type="application/dicom",
        patient_id="TEST123",
        patient_name="Test Patient",
        batch_instance_uid=batch_instance_uid,
        series_instance_uid="1.2.3.4.5.6.7.8.9.10",
        sop_instance_uid="1.2.3.4.5.6.7.8.9.10.11",
        modality="CT",
        batch_date=datetime.now().date(),
        dicom_metadata={},
    )
    db.add(dicom_file)
    db.commit()
    db.refresh(dicom_file)
    return dicom_file


# Pytest fixtures for common test setup
def pytest_setup_organization():
    """
    Pytest fixture for setting up a test organization.

    Usage:
        def test_something(pytest_setup_organization):
            org, admin, db = pytest_setup_organization
            # Use org and admin in your test

    Returns:
        Tuple of (organization, admin_user, db_session)
    """
    db = next(get_db())
    org, admin = setup_test_organization(db)

    yield org, admin, db

    # Cleanup
    teardown_test_organization(db, org.slug)
