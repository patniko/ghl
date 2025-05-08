import os
import io
from pytest import Session

from db import get_db
from models import User, Batch, File, ProcessingStatus, FileType, Organization
from auth import create_access_token

from tests.test_factory import create_test_client

# Create test client and ensure test organization exists
client = create_test_client()

# Run create_test_client again to ensure test organization exists
# This is needed because the client creation and organization setup are separate operations
create_test_client()


# Helper functions for file tests
def setup_test_user_and_org(db: Session):
    """Create a test user and organization for file tests"""
    # Generate unique email and phone to avoid conflicts
    import uuid

    unique_id = str(uuid.uuid4())[:8]

    # First clean up any existing test users to avoid conflicts
    try:
        users = (
            db.query(User)
            .filter(
                (User.email.like("test-%@example.com")) | (User.phone.like("+1123456%"))
            )
            .all()
        )

        for user in users:
            db.delete(user)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error cleaning up test users: {str(e)}")

    # Get test organization
    org = db.query(Organization).filter(Organization.slug == "test").first()
    if not org:
        raise Exception("Test organization not found. Run create_test_client() first.")

    # Create new user with unique identifiers
    user = User(
        first_name="Test",
        last_name="User",
        email=f"test-{unique_id}@example.com",
        email_verified=False,
        picture="",
        is_admin=True,
    )
    db.add(user)
    db.flush()  # Flush to get the user ID

    # Create user-organization relationship
    from models import UserOrganization

    user_org = UserOrganization(user_id=user.id, organization_id=org.id, is_admin=True)
    db.add(user_org)
    db.commit()
    db.refresh(user)
    db.refresh(org)
    return user, org


def teardown_test_user_and_org(db: Session, user_id: int, org_id: int):
    """Clean up test user, organization, and associated data"""
    try:
        # First rollback any pending transactions
        db.rollback()

        # Delete files first
        files = db.query(File).filter(File.user_id == user_id).all()
        for file in files:
            # Delete physical file if it exists
            if os.path.exists(file.file_path):
                try:
                    os.remove(file.file_path)
                except Exception as e:
                    print(f"Error removing file {file.file_path}: {str(e)}")
            db.delete(file)
        db.flush()

        # Delete batches
        batches = db.query(Batch).filter(Batch.user_id == user_id).all()
        for batch in batches:
            db.delete(batch)
        db.flush()

        # Delete user-organization relationships
        from models import UserOrganization

        user_orgs = (
            db.query(UserOrganization).filter(UserOrganization.user_id == user_id).all()
        )
        for user_org in user_orgs:
            db.delete(user_org)
        db.flush()

        # Delete the user
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            db.delete(user)
        db.flush()

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in teardown: {str(e)}")
        # Try one more time with a simpler approach
        try:
            db.execute(f"DELETE FROM files WHERE user_id = {user_id}")
            db.execute(f"DELETE FROM batches WHERE user_id = {user_id}")
            db.execute(f"DELETE FROM user_organizations WHERE user_id = {user_id}")
            db.execute(f"DELETE FROM users WHERE id = {user_id}")
            db.commit()
        except Exception as e2:
            db.rollback()
            print(f"Error in fallback teardown: {str(e2)}")


def create_test_batch(db: Session, user_id: int, org_id: int, name: str = "Test Batch"):
    """Create a test batch for the specified user and organization"""
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
    db: Session, user_id: int, org_id: int, batch_id: int, file_type: str = FileType.CSV
):
    """Create a test file for the specified batch"""
    file = File(
        organization_id=org_id,
        user_id=user_id,
        batch_id=batch_id,
        filename="test.csv" if file_type == FileType.CSV else f"test.{file_type}",
        original_filename="test.csv"
        if file_type == FileType.CSV
        else f"test.{file_type}",
        file_path=f"/uploads/projects/00/batches/{batch_id}/test.{file_type}",
        file_size=1024,
        content_type="text/csv"
        if file_type == FileType.CSV
        else f"application/{file_type}",
        file_type=file_type,
        file_metadata={},
        csv_headers=["col1", "col2", "col3"] if file_type == FileType.CSV else None,
        processing_status=ProcessingStatus.COMPLETED,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def get_auth_token(user_id: int, org_id: int = None):
    """Create an authentication token for the test user"""
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


def create_test_csv_content():
    """Create a test CSV file content"""
    return "header1,header2,header3\nvalue1,value2,value3\nvalue4,value5,value6"


# File Tests
# def test_upload_file_with_custom_type():
#     """Test uploading a file with a custom file type"""
#     db = next(get_db())
#     try:
#         # Setup test user and batch
#         user = setup_test_user(db)
#         token = get_auth_token(user.id)
#         batch = create_test_batch(db, user.id)

#         # Create test file content
#         file_content = io.BytesIO(b"test content")
#         file_content.name = "test.dat"

#         # Upload file with custom type
#         response = client.post(
#             "/files/upload",
#             files={"file": ("test.dat", file_content, "application/octet-stream")},
#             data={"batch_id": batch.id, "file_type": "custom_type"},
#             headers={"Authorization": f"Bearer {token}"},
#         )

#         assert response.status_code == 200
#         data = response.json()
#         assert data["original_filename"] == "test.dat"
#         assert data["file_type"] == "custom_type"  # Should use the provided type
#         assert data["batch_id"] == batch.id

#     finally:
#         teardown_test_user(db, user.id)


def test_upload_file_nonexistent_batch():
    """Test uploading a file to a non-existent batch"""
    db = next(get_db())
    try:
        # Setup test user and organization
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)

        # Create test file content
        file_content = io.BytesIO(b"test content")
        file_content.name = "test.txt"

        # Upload file to non-existent batch
        response = client.post(
            f"/{org.slug}/files/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
            data={"batch_id": 9999},  # Non-existent batch ID
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Batch not found"

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_get_files():
    """Test retrieving all files for a user"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch1 = create_test_batch(db, user.id, org.id, "Batch 1")
        batch2 = create_test_batch(db, user.id, org.id, "Batch 2")

        # Create multiple files
        file1 = create_test_file(db, user.id, org.id, batch1.id, FileType.CSV)
        file2 = create_test_file(db, user.id, org.id, batch2.id, FileType.MP4)

        # Get all files
        response = client.get(
            f"/{org.slug}/files/", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert any(f["id"] == file1.id for f in data)
        assert any(f["id"] == file2.id for f in data)

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_get_files_with_batch_filter():
    """Test retrieving files with batch filter"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch1 = create_test_batch(db, user.id, org.id, "Batch 1")
        batch2 = create_test_batch(db, user.id, org.id, "Batch 2")

        # Create multiple files
        file1 = create_test_file(db, user.id, org.id, batch1.id)
        _ = create_test_file(db, user.id, org.id, batch2.id)

        # Get files for batch1
        response = client.get(
            f"/{org.slug}/files/?batch_id={batch1.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == file1.id
        assert data[0]["batch_id"] == batch1.id

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_get_files_with_type_filter():
    """Test retrieving files with type filter"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch = create_test_batch(db, user.id, org.id)

        # Create files of different types
        file1 = create_test_file(db, user.id, org.id, batch.id, FileType.CSV)
        _ = create_test_file(db, user.id, org.id, batch.id, FileType.MP4)

        # Get CSV files
        response = client.get(
            f"/{org.slug}/files/?file_type={FileType.CSV}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == file1.id
        assert data[0]["file_type"] == FileType.CSV

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_get_file():
    """Test retrieving a specific file"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch = create_test_batch(db, user.id, org.id)
        file = create_test_file(db, user.id, org.id, batch.id)

        # Get file
        response = client.get(
            f"/{org.slug}/files/{file.id}", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == file.id
        assert data["original_filename"] == file.original_filename
        assert data["file_type"] == file.file_type
        assert data["batch_id"] == batch.id

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_get_nonexistent_file():
    """Test retrieving a file that doesn't exist"""
    db = next(get_db())
    try:
        # Setup test user
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)

        # Get non-existent file
        response = client.get(
            f"/{org.slug}/files/9999", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "File not found"

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_update_file():
    """Test updating file metadata"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch = create_test_batch(db, user.id, org.id)
        file = create_test_file(db, user.id, org.id, batch.id)

        # Update file
        update_data = {
            "file_type": "custom_type",
            "file_metadata": {"key1": "value1", "key2": "value2"},
        }
        response = client.put(
            f"/{org.slug}/files/{file.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == file.id
        assert data["file_type"] == update_data["file_type"]
        assert data["file_metadata"] == update_data["file_metadata"]

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_update_nonexistent_file():
    """Test updating a file that doesn't exist"""
    db = next(get_db())
    try:
        # Setup test user
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)

        # Update non-existent file
        update_data = {"file_type": "custom_type", "file_metadata": {"key1": "value1"}}
        response = client.put(
            f"/{org.slug}/files/9999",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "File not found"

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_delete_file():
    """Test deleting a file"""
    db = next(get_db())
    try:
        # Setup test user and data
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)
        batch = create_test_batch(db, user.id, org.id)

        # Use a temporary file instead of trying to create in /uploads
        import tempfile

        temp_dir = tempfile.mkdtemp()
        test_file_path = os.path.join(temp_dir, "test.csv")

        with open(test_file_path, "w") as f:
            f.write("test content")

        # Create file record with the temp file path
        file = File(
            organization_id=org.id,
            user_id=user.id,
            batch_id=batch.id,
            filename="test.csv",
            original_filename="test.csv",
            file_path=test_file_path,
            file_size=len("test content"),
            content_type="text/csv",
            file_type=FileType.CSV,
            file_metadata={},
            csv_headers=["col1", "col2", "col3"],
            processing_status=ProcessingStatus.COMPLETED,
        )
        db.add(file)
        db.commit()
        db.refresh(file)

        # Delete file
        response = client.delete(
            f"/{org.slug}/files/{file.id}", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        assert response.json()["message"] == "File deleted successfully"

        # Verify file is deleted from database
        db_file = db.query(File).filter(File.id == file.id).first()
        assert db_file is None

        # Verify physical file is deleted
        assert not os.path.exists(test_file_path)

    finally:
        teardown_test_user_and_org(db, user.id, org.id)
        # Clean up temp directory if it still exists
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)


def test_delete_nonexistent_file():
    """Test deleting a file that doesn't exist"""
    db = next(get_db())
    try:
        # Setup test user
        user, org = setup_test_user_and_org(db)
        token = get_auth_token(user.id, org.id)

        # Delete non-existent file
        response = client.delete(
            f"/{org.slug}/files/9999", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "File not found"

    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_unauthorized_access():
    """Test unauthorized access to file endpoints"""
    db = next(get_db())
    try:
        # Setup test user and organization (just to get the slug)
        user, org = setup_test_user_and_org(db)

        # Try to access files without authentication
        response = client.get(f"/{org.slug}/files/")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

        # Try to upload a file without authentication
        file_content = io.BytesIO(b"test content")
        response = client.post(
            f"/{org.slug}/files/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
            data={"batch_id": 1},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"
    finally:
        teardown_test_user_and_org(db, user.id, org.id)


def test_detect_file_type():
    """Test file type detection functionality"""
    from services.files import detect_file_type

    # Test various file extensions
    assert detect_file_type("test.csv") == FileType.CSV
    assert detect_file_type("test.dcm") == FileType.DICOM
    assert detect_file_type("test.dicom") == FileType.DICOM
    assert detect_file_type("test.mp4") == FileType.MP4
    assert detect_file_type("test.m4v") == FileType.MP4
    assert detect_file_type("test.npz") == FileType.NPZ

    # Test unknown extension
    assert detect_file_type("test.xyz") == "xyz"

    # Test no extension
    assert detect_file_type("test") == ""


def test_extract_csv_headers():
    """Test CSV header extraction functionality"""
    from services.files import extract_csv_headers
    import tempfile

    # Create a temporary CSV file in a directory we have write access to
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "test_headers.csv")

    try:
        # Write test data
        with open(temp_path, "w") as f:
            f.write("header1,header2,header3\nvalue1,value2,value3")

        # Test header extraction
        headers = extract_csv_headers(temp_path)
        assert headers == ["header1", "header2", "header3"]

        # Test with empty file
        with open(temp_path, "w") as f:
            f.write("")

        # Should return empty list for empty file
        headers = extract_csv_headers(temp_path)
        assert headers == []

    finally:
        # Clean up
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
