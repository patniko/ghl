from db import get_db
from tests.test_factory import create_test_client
from tests.test_utils import (
    get_auth_token,
    setup_test_organization,
    teardown_test_organization,
    create_test_model,
    create_test_batch,
    create_test_file,
)

client = create_test_client()


# API Tests
def test_create_model():
    """Test creating a new model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create model
        model_data = {
            "name": "Test Text Classification Model",
            "description": "A model for classifying text",
            "version": "1.0.0",
            "parameters": {"threshold": 0.5, "max_length": 512},
            "implementation": "text_classification",
        }
        response = client.post(
            f"/{org.slug}/catalog/models",
            json=model_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == model_data["name"]
        assert data["description"] == model_data["description"]
        assert data["version"] == model_data["version"]
        assert data["parameters"] == model_data["parameters"]
        assert data["implementation"] == model_data["implementation"]
        assert data["is_system"] is False

    except Exception as e:
        print(f"Error in test_create_model: {str(e)}")
        raise
    finally:
        teardown_test_organization(db, org.slug)


def test_create_model_invalid_implementation():
    """Test creating a model with an invalid implementation"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create model with invalid implementation
        model_data = {
            "name": "Invalid Model",
            "description": "This model has an invalid implementation",
            "version": "1.0.0",
            "parameters": {},
            "implementation": "nonexistent_implementation",
        }
        response = client.post(
            f"/{org.slug}/catalog/models",
            json=model_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400
        assert (
            "Implementation 'nonexistent_implementation' not found"
            in response.json()["detail"]
        )

    finally:
        teardown_test_organization(db, org.slug)


def test_get_models():
    """Test retrieving all models"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create multiple models
        model1 = create_test_model(
            db, org.id, "Text Classification Model", "text_classification"
        )
        model2 = create_test_model(
            db, org.id, "Image Classification Model", "image_classification"
        )

        # Get all models
        response = client.get(
            f"/{org.slug}/catalog/models", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2  # There might be system models already
        assert any(m["id"] == model1.id for m in data)
        assert any(m["id"] == model2.id for m in data)

    finally:
        teardown_test_organization(db, org.slug)


def test_get_models_with_filters():
    """Test retrieving models with filters"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create models of different types
        model1 = create_test_model(
            db, org.id, "Text Classification Model", "text_classification"
        )
        model2 = create_test_model(
            db, org.id, "Image Classification Model", "image_classification"
        )
        model3 = create_test_model(
            db, org.id, "System Model", "text_classification", is_system=True
        )

        # Get models with text_classification implementation
        response = client.get(
            f"/{org.slug}/catalog/models?implementation=text_classification",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert any(m["id"] == model1.id for m in data)
        assert not any(m["id"] == model2.id for m in data)
        assert any(m["id"] == model3.id for m in data)

        # Get system models
        response = client.get(
            f"/{org.slug}/catalog/models?is_system=true",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert not any(m["id"] == model1.id for m in data)
        assert not any(m["id"] == model2.id for m in data)
        assert any(m["id"] == model3.id for m in data)

    finally:
        teardown_test_organization(db, org.slug)


def test_get_model():
    """Test retrieving a specific model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create model
        model = create_test_model(db, org.id, "Test Model", "text_classification")

        # Get model
        response = client.get(
            f"/{org.slug}/catalog/models/{model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model.id
        assert data["name"] == model.name
        assert data["implementation"] == model.implementation
        assert data["version"] == model.version

    finally:
        teardown_test_organization(db, org.slug)


def test_get_nonexistent_model():
    """Test retrieving a model that doesn't exist"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Get non-existent model
        response = client.get(
            f"/{org.slug}/catalog/models/9999",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Model not found"

    finally:
        teardown_test_organization(db, org.slug)


def test_update_model():
    """Test updating a model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create model
        model = create_test_model(db, org.id, "Test Model", "text_classification")

        # Update model
        update_data = {
            "name": "Updated Model Name",
            "description": "Updated model description",
            "version": "1.1.0",
            "parameters": {"new_param": "value"},
        }
        response = client.put(
            f"/{org.slug}/catalog/models/{model.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model.id
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["version"] == update_data["version"]
        assert data["parameters"] == update_data["parameters"]
        assert data["implementation"] == model.implementation  # Unchanged

    finally:
        teardown_test_organization(db, org.slug)


def test_update_system_model():
    """Test updating a system model (should be forbidden)"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create system model
        model = create_test_model(
            db,
            org.id,
            "System Model",
            "text_classification",
            is_system=True,
        )

        # Try to update system model
        update_data = {
            "name": "Updated System Model",
            "description": "This update should fail",
        }
        response = client.put(
            f"/{org.slug}/catalog/models/{model.id}",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "System models cannot be modified"

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_model():
    """Test deleting a model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create model
        model = create_test_model(db, org.id, "Test Model", "text_classification")

        # Delete model
        response = client.delete(
            f"/{org.slug}/catalog/models/{model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Model deleted successfully"

        # Verify model is deleted
        response = client.get(
            f"/{org.slug}/catalog/models/{model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    finally:
        teardown_test_organization(db, org.slug)


def test_delete_system_model():
    """Test deleting a system model (should be forbidden)"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create system model
        model = create_test_model(
            db,
            org.id,
            "System Model",
            "text_classification",
            is_system=True,
        )

        # Try to delete system model
        response = client.delete(
            f"/{org.slug}/catalog/models/{model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "System models cannot be deleted"

    finally:
        teardown_test_organization(db, org.slug)


def test_evaluate_file():
    """Test evaluating a file with a model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a batch
        batch = create_test_batch(db, admin.id, org.id)

        # Create model
        model = create_test_model(
            db, org.id, "Text Classification Model", "text_classification"
        )

        # Create file
        file = create_test_file(db, admin.id, org.id, batch.id, "csv")

        # Evaluate file
        response = client.post(
            f"/{org.slug}/catalog/models/evaluate-file/{file.id}?model_id={model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["compatible"] is True
        assert data["file_id"] == file.id
        assert data["model_id"] == model.id
        assert "evaluation_results" in data
        assert "model_name" in data["evaluation_results"]
        assert "results" in data["evaluation_results"]
        assert "predictions" in data["evaluation_results"]["results"]

        # Check that file metadata was updated
        db.refresh(file)
        assert "model_evaluations" in file.file_metadata
        assert len(file.file_metadata["model_evaluations"]) == 1
        assert file.file_metadata["model_evaluations"][0]["model_id"] == model.id

    finally:
        teardown_test_organization(db, org.slug)


def test_evaluate_file_incompatible():
    """Test evaluating a file that's incompatible with the model"""
    db = next(get_db())
    org, admin = setup_test_organization(db)

    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Create a batch
        batch = create_test_batch(db, admin.id, org.id)

        # Create model for image classification
        model = create_test_model(
            db, org.id, "Image Classification Model", "image_classification"
        )

        # Create CSV file (incompatible with image classification)
        file = create_test_file(db, admin.id, org.id, batch.id, "csv")

        # Evaluate file
        response = client.post(
            f"/{org.slug}/catalog/models/evaluate-file/{file.id}?model_id={model.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["compatible"] is False
        assert "reason" in data
        assert data["file_id"] == file.id
        assert data["model_id"] == model.id

    finally:
        teardown_test_organization(db, org.slug)
