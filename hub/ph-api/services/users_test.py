from datetime import datetime
import io

from PIL import Image
from pytest import Session

from db import get_db
from models import User, Organization

from tests.test_factory import create_test_client

# Create test client and ensure test organization exists
client = create_test_client()

# Run create_test_client again to ensure test organization exists
# This is needed because the client creation and organization setup are separate operations
create_test_client()


# Helper functions for user tests
def setup_test_account(email: str = None):
    db = next(get_db())

    # Generate unique email to avoid conflicts
    import uuid

    unique_id = str(uuid.uuid4())[:8]
    if email is None:
        email = f"test-{unique_id}@example.com"

    # First clean up any existing test users to avoid conflicts
    try:
        users = db.query(User).filter(User.email.like("test-%@example.com")).all()

        for user in users:
            db.delete(user)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error cleaning up test users: {str(e)}")

    # Get test organization
    test_org = db.query(Organization).filter(Organization.slug == "test").first()
    if not test_org:
        raise Exception("Test organization not found. Run create_test_client() first.")

    return email, test_org.id


def teardown_test_account(email: str):
    db = next(get_db())
    try:
        # First rollback any pending transactions
        db.rollback()

        # Find the user
        user = db.query(User).filter(User.email == email).first()
        if user:
            # Delete refresh tokens
            db.execute(f"DELETE FROM refresh_tokens WHERE user_id = {user.id}")
            # Delete the user
            db.delete(user)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in teardown: {str(e)}")
        # Try direct SQL approach
        try:
            user = db.query(User).filter(User.email == email).first()
            if user:
                db.execute(f"DELETE FROM refresh_tokens WHERE user_id = {user.id}")
                db.execute(f"DELETE FROM users WHERE id = {user.id}")
                db.commit()
        except Exception as e2:
            db.rollback()
            print(f"Error in fallback teardown: {str(e2)}")


def create_test_user(db: Session, email: str, organization_id: int = None):
    user = User(
        first_name="Test",
        last_name="User",
        email=email,
        email_verified=True,
        picture="",
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # If organization_id is provided, create a user-organization relationship
    if organization_id:
        from models import UserOrganization

        user_org = UserOrganization(
            user_id=user.id, organization_id=organization_id, is_admin=False
        )
        db.add(user_org)
        db.commit()

    return user


def create_test_image():
    # Create a small test image
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


# Email and Password Authentication Tests


# User Information Tests
def test_get_user_me():
    email = "test-get-me@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get a token
        login_data = {"email": email, "password": password}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Get user information
        response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == email
        assert "first_name" in data
        assert "last_name" in data
        assert "email_verified" in data
        assert "picture" in data
        assert "is_onboarded" in data
        assert "organizations" in data
    finally:
        teardown_test_account(email)


def test_get_user_me_unauthorized():
    response = client.get("/users/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# User Update Tests
def test_update_user_profile():
    email = "test-update@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get a token
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Update user profile
        update_data = {
            "first_name": "Test",
            "last_name": "User",
            "picture": "https://example.com/pic.jpg",
        }
        response = client.put(
            "/users/me",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["first_name"] == update_data["first_name"]
        assert data["last_name"] == update_data["last_name"]
        assert data["picture"] == update_data["picture"]
        assert data["is_onboarded"] is True
    finally:
        teardown_test_account(email)


def test_update_user_partial_profile_name():
    email = "test-update-partial@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get a token
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Update user profile
        update_data = {"first_name": "Test", "last_name": "User", "picture": None}
        response = client.put(
            "/users/me",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["first_name"] == update_data["first_name"]
        assert data["last_name"] == update_data["last_name"]
        assert data["picture"] == ""
        assert data["is_onboarded"] is True
    finally:
        teardown_test_account(email)


def test_update_user_profile_unauthorized():
    update_data = {"first_name": "Test", "last_name": "User"}
    response = client.put("/users/me", json=update_data)
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_get_avatar():
    email = "test-avatar@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get user ID
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        user_id = login_response.json()["user_id"]

        # Get avatar
        response = client.get(f"/users/avatar/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "first_name" in data
        assert "last_initial" in data
        assert "picture" in data
    finally:
        teardown_test_account(email)


def test_get_avatar_nonexistent_user():
    response = client.get("/users/avatar/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "No user found."


# Token Tests
def test_get_token():
    email = "test-token@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get initial token
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Get new token
        response = client.get(
            "/users/me/token", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "user_id" in data
        assert data["token_type"] == "bearer"
    finally:
        teardown_test_account(email)


def test_get_token_unauthorized():
    response = client.get("/users/me/token")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# Password Tests
def test_set_password():
    email = "test-set-password@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user without a password
        db = next(get_db())
        user = create_test_user(db, email, org_id)
        db.commit()

        # Create a token for the user
        from auth import create_access_token

        token = create_access_token(
            data={
                "sub": str(user.id),
                "user_id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "email_verified": user.email_verified,
                "picture": user.picture,
                "updated_at": datetime.now().isoformat() + "Z",
                "admin": False,
                "password_set": False,
                "is_admin": user.is_admin,
            }
        )

        # Set initial password
        password_data = {"new_password": "testpassword123"}
        response = client.post(
            "/users/password/set",
            json=password_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Get the new token
        token = data["access_token"]

        # Try to set new password without providing current password
        new_password_data = {"new_password": "newpassword123"}
        response = client.post(
            "/users/password/set",
            json=new_password_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

        # Set new password with current password
        update_data = {
            "current_password": "testpassword123",
            "new_password": "newpassword123",
        }
        response = client.post(
            "/users/password/set",
            json=update_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    finally:
        teardown_test_account(email)


def test_password_login():
    email = "test-login@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Test successful login
        login_data = {"email": email, "password": password, "organization_id": org_id}
        response = client.post("/users/password/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Test invalid password
        login_data = {
            "email": email,
            "password": "wrongpassword",
            "organization_id": org_id,
        }
        response = client.post("/users/password/login", json=login_data)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

        # Test non-existent user
        login_data = {
            "email": "nonexistent@example.com",
            "password": password,
            "organization_id": org_id,
        }
        response = client.post("/users/password/login", json=login_data)
        assert response.status_code == 404
        assert response.json()["detail"] == "User not found"
    finally:
        teardown_test_account(email)


def test_password_login_not_enabled():
    email = "test-login-not-enabled@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user without a password
        db = next(get_db())
        _ = create_test_user(db, email, org_id)
        db.commit()

        # Try to login with password before setting one
        login_data = {
            "email": email,
            "password": "somepassword",
            "organization_id": org_id,
        }
        response = client.post("/users/password/login", json=login_data)
        assert response.status_code == 400
        assert response.json()["detail"] == "Password login not enabled for this user"
    finally:
        teardown_test_account(email)


# Account Deletion Test
def test_delete_account():
    email = "test-delete@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get a token
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        user_id = login_response.json()["user_id"]

        # Delete account
        response = client.delete(
            "/users/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Account successfully deleted"

        # Verify account is marked as deleted
        db = next(get_db())
        user = db.query(User).filter(User.id == user_id).first()
        assert user.is_deleted is True
        assert user.deleted_at is not None
    finally:
        teardown_test_account(email)


# Refresh Token Tests
def test_refresh_token():
    email = "test-refresh@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get tokens
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        data = login_response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        refresh_token = data["refresh_token"]

        # Use refresh token to get new access token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/users/refresh", json=refresh_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["refresh_token"] == refresh_token  # Same refresh token returned
        assert "token_type" in data
        assert "user_id" in data
        assert "expires_in" in data
    finally:
        teardown_test_account(email)


def test_refresh_token_invalid():
    # Try to refresh with invalid token
    refresh_data = {"refresh_token": "invalid_token"}
    response = client.post("/users/refresh", json=refresh_data)
    assert response.status_code == 401
    assert "detail" in response.json()


def test_revoke_token():
    email = "test-revoke@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get tokens
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        data = login_response.json()
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]

        # First use the refresh token to ensure it's valid
        refresh_data = {"refresh_token": refresh_token}
        refresh_response = client.post("/users/refresh", json=refresh_data)
        assert refresh_response.status_code == 200

        # Now revoke the refresh token
        revoke_data = {"refresh_token": refresh_token}
        response = client.post(
            "/users/revoke",
            json=revoke_data,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert "message" in response.json()

        # Try to use the revoked refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/users/refresh", json=refresh_data)
        assert response.status_code == 401
    finally:
        teardown_test_account(email)


def test_revoke_all_tokens():
    email = "test-revoke-all@example.com"
    email, org_id = setup_test_account(email)

    try:
        # Create a user with a password
        db = next(get_db())
        password = "testpassword123"
        user = create_test_user(db, email, org_id)

        # Set password for the user
        from auth import get_password_hash

        user.password_hash = get_password_hash(password)
        db.commit()

        # Login to get tokens
        login_data = {"email": email, "password": password, "organization_id": org_id}
        login_response = client.post("/users/password/login", json=login_data)
        assert login_response.status_code == 200
        data = login_response.json()
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]

        # Revoke all tokens for the user
        response = client.post(
            "/users/revoke-all",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert "message" in response.json()

        # Try to use the revoked refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/users/refresh", json=refresh_data)
        assert response.status_code == 401
    finally:
        teardown_test_account(email)
