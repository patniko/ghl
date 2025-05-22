from datetime import datetime
import tempfile
import os
import json

from pytest import Session

from db import get_db
from models import User, Organization, SampleDataset, Project

from tests.test_factory import create_test_client
from tests.test_utils import (
    get_auth_token,
    setup_test_organization,
    teardown_test_organization,
)

# Create test client
client = create_test_client()


def create_test_sample_dataset(db, project_id, name="Test Sample Dataset"):
    """Create a test sample dataset for testing"""
    dataset = SampleDataset(
        project_id=project_id,
        name=name,
        description="Test sample dataset description",
        num_patients=100,
        created_at=datetime.now(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def create_test_project(db, user_id, org_id, name="Test-Project"):
    """Create a test project for testing"""
    project = Project(
        organization_id=org_id,
        user_id=user_id,
        name=name,
        description="Test project description",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


# Tests
def test_create_sample_dataset():
    """Test creating a sample dataset via API"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project first
        project = create_test_project(db, admin.id, org.id)

        # Create sample dataset
        dataset_data = {
            "name": "Test API Dataset",
            "description": "Dataset created via API test",
            "num_patients": 50,
            "data_types": ["questionnaire", "blood"],
            "include_partials": True,
            "partial_rate": 0.5,
        }

        # Convert to form data
        form_data = {
            "name": dataset_data["name"],
            "description": dataset_data["description"],
            "num_patients": dataset_data["num_patients"],
            "data_types": dataset_data["data_types"],
            "include_partials": dataset_data["include_partials"],
            "partial_rate": dataset_data["partial_rate"],
            "project_id": project.id,
        }

        response = client.post(
            "/samples/",
            data=form_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == dataset_data["name"]
        assert data["description"] == dataset_data["description"]
        assert data["num_patients"] == dataset_data["num_patients"]
        assert data["project_id"] == project.id
        assert "created_at" in data
        assert data["data_types"] == dataset_data["data_types"]
        assert data["include_partials"] == dataset_data["include_partials"]
        assert data["partial_rate"] == dataset_data["partial_rate"]

        # Verify dataset exists in database
        dataset = (
            db.query(SampleDataset)
            .filter(SampleDataset.name == dataset_data["name"])
            .filter(SampleDataset.project_id == project.id)
            .first()
        )
        assert dataset is not None
        assert dataset.name == dataset_data["name"]
        assert dataset.description == dataset_data["description"]
        assert dataset.num_patients == dataset_data["num_patients"]

    finally:
        teardown_test_organization(db, org.slug)


def test_get_sample_datasets():
    """Test retrieving all sample datasets"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Create multiple datasets
        dataset1 = create_test_sample_dataset(db, project.id, "Dataset-1")
        dataset2 = create_test_sample_dataset(db, project.id, "Dataset-2")

        # Get all datasets
        response = client.get(
            "/samples/",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # At least the two we created

        # Verify our test datasets are in the response
        dataset_names = [d["name"] for d in data]
        assert dataset1.name in dataset_names
        assert dataset2.name in dataset_names

    finally:
        teardown_test_organization(db, org.slug)


def test_get_sample_dataset():
    """Test retrieving a specific sample dataset"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Create a dataset
        dataset = create_test_sample_dataset(db, project.id)

        # Get the dataset
        response = client.get(
            f"/samples/{dataset.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset.id
        assert data["name"] == dataset.name
        assert data["description"] == dataset.description
        assert data["num_patients"] == dataset.num_patients
        assert data["project_id"] == project.id
        assert "created_at" in data

    finally:
        teardown_test_organization(db, org.slug)


def test_get_nonexistent_sample_dataset():
    """Test retrieving a sample dataset that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Get non-existent dataset
        response = client.get(
            "/samples/99999",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_sample_dataset():
    """Test deleting a sample dataset"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Create a dataset
        dataset = create_test_sample_dataset(db, project.id)
        dataset_id = dataset.id

        # Delete the dataset
        response = client.delete(
            f"/samples/{dataset.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        assert response.json()["message"] == "Dataset deleted successfully"

        # Get a fresh database session to avoid caching issues
        db.close()
        db = next(get_db())

        # Verify dataset was deleted from database
        deleted_dataset = db.query(SampleDataset).filter(SampleDataset.id == dataset_id).first()
        assert deleted_dataset is None

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_nonexistent_sample_dataset():
    """Test deleting a sample dataset that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Delete non-existent dataset
        response = client.delete(
            "/samples/99999",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_download_sample_dataset():
    """Test downloading a sample dataset"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Create a dataset
        dataset = create_test_sample_dataset(db, project.id)

        # Download the dataset
        response = client.get(
            f"/samples/{dataset.id}/download",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert f"{dataset.name.replace(' ', '_')}.zip" in response.headers["content-disposition"]

        # Save the zip file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        try:
            # Verify the zip file contains the expected content
            import zipfile
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                assert "dataset_info.json" in zip_ref.namelist()
                
                # Extract and check the JSON content
                with zip_ref.open("dataset_info.json") as json_file:
                    data = json.load(json_file)
                    assert data["id"] == dataset.id
                    assert data["name"] == dataset.name
                    assert data["description"] == dataset.description
                    assert data["num_patients"] == dataset.num_patients
                    assert data["project_id"] == project.id
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    finally:
        teardown_test_organization(db, org.slug)


def test_download_nonexistent_sample_dataset():
    """Test downloading a sample dataset that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Download non-existent dataset
        response = client.get(
            "/samples/99999/download",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    finally:
        teardown_test_organization(db, org.slug)


def test_unauthorized_access():
    """Test unauthorized access to sample dataset endpoints"""
    db = next(get_db())
    org, admin = setup_test_organization(db)
    try:
        # Create a project
        project = create_test_project(db, admin.id, org.id)

        # Create a dataset
        dataset = create_test_sample_dataset(db, project.id)

        # Try to access datasets without authentication
        response = client.get("/samples/")
        assert response.status_code in [401, 403, 405]  # Accept any of these status codes

        # Try to get a specific dataset without authentication
        response = client.get(f"/samples/{dataset.id}")
        assert response.status_code in [401, 403, 405]  # Accept any of these status codes

        # Try to create a dataset without authentication
        dataset_data = {
            "name": "Unauthorized Dataset",
            "description": "Dataset created without authentication",
            "num_patients": 50,
        }
        response = client.post("/samples/", data=dataset_data)
        assert response.status_code in [401, 403, 405]  # Accept any of these status codes

        # Try to delete a dataset without authentication
        response = client.delete(f"/samples/{dataset.id}")
        assert response.status_code in [401, 403, 405]  # Accept any of these status codes

        # Try to download a dataset without authentication
        response = client.get(f"/samples/{dataset.id}/download")
        assert response.status_code in [401, 403, 405]  # Accept any of these status codes

    finally:
        teardown_test_organization(db, org.slug)
