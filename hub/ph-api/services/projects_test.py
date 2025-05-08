from db import get_db
from models import Project, DataRegion
from tests.test_factory import create_test_client
from tests.test_utils import (
    get_auth_token,
    setup_test_organization,
    teardown_test_organization,
)

# Create test client
client = create_test_client()


def create_test_project(db, user_id, org_id, name="Test-Project"):
    """Create a test project for testing"""
    project = Project(
        organization_id=org_id,
        user_id=user_id,
        name=name,
        description="Test project description",
        data_region=DataRegion.LOCAL.value,
        s3_bucket_name="",  # Initialize with empty string instead of None
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


# Tests
def test_create_project():
    """Test creating a project via API"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create project
        project_data = {
            "name": "New-API-Project",  # Use hyphens instead of spaces to comply with validation
            "description": "Project created via API test",
            "data_region": DataRegion.LOCAL.value,
            "s3_bucket_name": "",  # Empty string instead of None
        }

        response = client.post(
            f"/projects/{org.slug}",
            json=project_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Print response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == project_data["name"]
        assert data["description"] == project_data["description"]
        assert data["data_region"] == project_data["data_region"]
        assert "created_at" in data
        assert "updated_at" in data

        # Verify project exists in database
        project = (
            db.query(Project)
            .filter(Project.name == project_data["name"])
            .filter(Project.organization_id == org.id)
            .first()
        )
        assert project is not None
        assert project.name == project_data["name"]
        assert project.description == project_data["description"]
        assert project.data_region == project_data["data_region"]

    finally:
        teardown_test_organization(db, org.slug)


def test_create_project_invalid_data_region():
    """Test creating a project with invalid data_region"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create project with invalid data_region
        project_data = {
            "name": "Invalid-Project",  # Use hyphens instead of spaces to comply with validation
            "description": "Project with invalid data_region",
            "data_region": "invalid_region",
            "s3_bucket_name": "",  # Empty string instead of None
        }

        response = client.post(
            f"/projects/{org.slug}",
            json=project_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 400
        assert "Invalid data_region" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_get_projects():
    """Test retrieving all projects for a user"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create multiple projects
        project1 = create_test_project(db, admin.id, org.id, "Project-1")
        project2 = create_test_project(db, admin.id, org.id, "Project-2")

        # Get all projects
        response = client.get(
            f"/projects/{org.slug}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # At least the two we created

        # Verify our test projects are in the response
        project_names = [p["name"] for p in data]
        assert project1.name in project_names
        assert project2.name in project_names

    finally:
        teardown_test_organization(db, org.slug)


def test_get_project():
    """Test retrieving a specific project"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Get the project
        response = client.get(
            f"/projects/{org.slug}/{project.name}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project.id
        assert data["name"] == project.name
        assert data["description"] == project.description
        assert data["data_region"] == project.data_region
        assert "created_at" in data
        assert "updated_at" in data

    finally:
        teardown_test_organization(db, org.slug)


def test_get_nonexistent_project():
    """Test retrieving a project that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Get non-existent project
        response = client.get(
            f"/projects/{org.slug}/nonexistent-project",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "Project not found" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_update_project():
    """Test updating a project"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)
        project_id = project.id

        # Print initial project state for debugging
        print(
            f"Initial project: id={project.id}, name={project.name}, data_region={project.data_region}"
        )

        # Update the project
        update_data = {
            "name": "Updated-Project",  # Use hyphens instead of spaces to comply with validation
            "description": "Updated description",
            "data_region": DataRegion.US.value,
            "s3_bucket_name": "test-bucket",
        }

        response = client.put(
            f"/projects/{org.slug}/{project.name}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["data_region"] == update_data["data_region"]
        assert data["s3_bucket_name"] == update_data["s3_bucket_name"]

        # Get a fresh database session to avoid caching issues
        db.close()
        db = next(get_db())

        # Verify project was updated in database
        updated_project = db.query(Project).filter(Project.id == project_id).first()
        print(
            f"Updated project from DB: id={updated_project.id}, name={updated_project.name}, data_region={updated_project.data_region}"
        )

        assert updated_project.name == update_data["name"]
        assert updated_project.description == update_data["description"]
        assert updated_project.data_region == update_data["data_region"]
        assert updated_project.s3_bucket_name == update_data["s3_bucket_name"]

    finally:
        teardown_test_organization(db, org.slug)


def test_update_project_partial():
    """Test partial update of a project"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)
        project_id = project.id
        original_description = project.description
        original_data_region = project.data_region

        # Update only the name
        update_data = {
            "name": "Partially-Updated-Project",  # Use hyphens instead of spaces to comply with validation
        }

        response = client.put(
            f"/projects/{org.slug}/{project.name}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == update_data["name"]
        assert data["description"] == original_description  # Unchanged
        assert data["data_region"] == original_data_region  # Unchanged

        # Get a fresh database session to avoid caching issues
        db.close()
        db = next(get_db())

        # Verify project was updated in database
        updated_project = db.query(Project).filter(Project.id == project_id).first()
        assert updated_project.name == update_data["name"]
        assert updated_project.description == original_description
        assert updated_project.data_region == original_data_region

    finally:
        teardown_test_organization(db, org.slug)


def test_update_project_invalid_data_region():
    """Test updating a project with invalid data_region"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)
        project_id = project.id

        # Update with invalid data_region
        update_data = {
            "data_region": "invalid_region",
        }

        response = client.put(
            f"/projects/{org.slug}/{project.name}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 400
        assert "Invalid data_region" in response.json()["detail"]

        # Get a fresh database session to avoid caching issues
        db.close()
        db = next(get_db())

        # Verify project was not updated in database
        unchanged_project = db.query(Project).filter(Project.id == project_id).first()
        assert (
            unchanged_project.data_region == DataRegion.LOCAL.value
        )  # Should remain unchanged

    finally:
        teardown_test_organization(db, org.slug)


def test_update_nonexistent_project():
    """Test updating a project that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Update non-existent project
        update_data = {
            "name": "Updated-Non-existent-Project",  # Use hyphens instead of spaces to comply with validation
        }

        response = client.put(
            f"/projects/{org.slug}/nonexistent-project",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "Project not found" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_project():
    """Test deleting a project"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)
        project_id = project.id

        # Delete the project
        response = client.delete(
            f"/projects/{org.slug}/{project.name}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        assert response.json()["message"] == "Project deleted successfully"

        # Get a fresh database session to avoid caching issues
        db.close()
        db = next(get_db())

        # Verify project was deleted from database
        deleted_project = db.query(Project).filter(Project.id == project_id).first()
        assert deleted_project is None

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_nonexistent_project():
    """Test deleting a project that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Delete non-existent project
        response = client.delete(
            f"/projects/{org.slug}/nonexistent-project",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code in [404, 500]  # Accept either 404 or 500
        if response.status_code == 404:
            assert "Project not found" in response.json()["detail"]
        else:
            assert "Error retrieving project" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_unauthorized_access():
    """Test unauthorized access to project endpoints"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Try to access projects without authentication
        response = client.get(f"/projects/{org.slug}")
        assert response.status_code in [
            401,
            403,
            405,
        ]  # Accept any of these status codes

        # Try to get a specific project without authentication
        response = client.get(f"/projects/{org.slug}/{project.name}")
        assert response.status_code in [
            401,
            403,
            405,
        ]  # Accept any of these status codes

        # Try to create a project without authentication
        project_data = {
            "name": "Unauthorized-Project",  # Use hyphens instead of spaces to comply with validation
        }
        response = client.post(f"/projects/{org.slug}", json=project_data)
        assert response.status_code in [
            401,
            403,
            405,
        ]  # Accept any of these status codes

        # Try to update a project without authentication
        update_data = {
            "name": "Updated-Unauthorized-Project",  # Use hyphens instead of spaces to comply with validation
        }
        response = client.put(f"/projects/{org.slug}/{project.name}", json=update_data)
        assert response.status_code in [
            401,
            403,
            405,
        ]  # Accept any of these status codes

        # Try to delete a project without authentication
        response = client.delete(f"/projects/{org.slug}/{project.name}")
        assert response.status_code in [
            401,
            403,
            405,
        ]  # Accept any of these status codes

    finally:
        teardown_test_organization(db, org.slug)
