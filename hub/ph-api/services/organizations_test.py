from db import get_db
from models import Organization
from tests.test_factory import create_test_client
from tests.test_utils import (
    get_auth_token,
    setup_test_organization,
    teardown_test_organization,
)

client = create_test_client()


# Tests
def test_list_organizations():
    """Test listing organizations"""
    org, admin = setup_test_organization(get_db().__next__())
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # List organizations
        response = client.get("/orgs", headers={"Authorization": f"Bearer {token}"})
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify organization data
        found = False
        for org_data in data:
            if org_data["slug"] == org.slug:
                found = True
                assert org_data["name"] == org.name
                assert "created_at" in org_data
                assert "updated_at" in org_data

        assert found, "Test organization not found in response"
    finally:
        teardown_test_organization(get_db().__next__(), org.slug)


def test_get_organization_by_slug():
    """Test getting an organization by slug"""
    org, admin = setup_test_organization(get_db().__next__())
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Get organization by slug
        response = client.get(
            f"/orgs/{org.slug}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == org.id
        assert data["name"] == org.name
        assert data["slug"] == org.slug
        assert "created_at" in data
        assert "updated_at" in data
    finally:
        teardown_test_organization(get_db().__next__(), org.slug)


def test_get_organization_by_slug_not_found():
    """Test getting a non-existent organization"""
    org, admin = setup_test_organization(get_db().__next__())
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Get non-existent organization
        response = client.get(
            "/orgs/non-existent-org",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 404
        assert "detail" in response.json()
    finally:
        teardown_test_organization(get_db().__next__(), org.slug)


def test_create_organization():
    """Test creating a new organization"""
    db = get_db().__next__()

    # Clean up any existing organization with the
    org, admin = setup_test_organization(db)
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        new_org_slug = "new-test-org"
        teardown_test_organization(db, new_org_slug)

        # Create new organization
        new_org_data = {
            "name": "New Test Organization",
            "slug": new_org_slug,
        }

        response = client.post(
            "/orgs",
            json=new_org_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == new_org_data["name"]
        assert data["slug"] == new_org_data["slug"]
        assert "created_at" in data
        assert "updated_at" in data

        # Clean up the new organization
        teardown_test_organization(get_db().__next__(), new_org_data["slug"])
    finally:
        teardown_test_organization(get_db().__next__(), org.slug)


def test_create_organization_duplicate_slug():
    """Test creating an organization with a duplicate slug"""
    org, admin = setup_test_organization(get_db().__next__())
    try:
        # Get auth token
        token = get_auth_token(admin.id, org.id)

        # Try to create organization with same slug
        duplicate_org_data = {
            "name": "Duplicate Organization",
            "slug": org.slug,
            "description": "This should fail",
        }

        response = client.post(
            "/orgs",
            json=duplicate_org_data,
            headers={"Authorization": f"Bearer {token}"},
        )

        # Verify response
        assert response.status_code == 400
        assert "detail" in response.json()
        assert "already exists" in response.json()["detail"]
    finally:
        teardown_test_organization(get_db().__next__(), org.slug)


def test_delete_organization():
    """Test deleting an organization"""
    db = next(get_db())
    # Use a unique slug with timestamp to avoid conflicts
    import time

    unique_slug = f"org-to-delete-{int(time.time())}"
    org, admin = setup_test_organization(db, unique_slug)

    # Get auth token
    token = get_auth_token(admin.id, org.id)

    # Delete organization
    response = client.delete(
        f"/orgs/{org.slug}/{org.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Verify response
    assert response.status_code == 200
    assert "message" in response.json()

    # Verify organization is deleted
    deleted_org = db.query(Organization).filter(Organization.id == org.id).first()
    assert deleted_org is None
