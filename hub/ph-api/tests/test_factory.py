import base64
import json
from functools import lru_cache
from fastapi.testclient import TestClient
from httpx import Client
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from env import get_settings
from server import app
from db import get_db
from models import Organization, User
from auth import get_password_hash


class Route(BaseModel):
    id: str | None = None
    primary: bool | None = None
    production_url: str | None = None
    to: str | None = None


class Routes(BaseModel):
    routes: dict[str, Route] = {"route": Route()}


class PlatformTestSettings(BaseSettings):
    platform_routes: str | None = None


_settings = get_settings()


def create_test_client():
    """Create a test client with a test organization and admin user"""
    # Create test organization and admin user if they don't exist
    db = next(get_db())
    try:
        # Check if test organization exists
        test_org = db.query(Organization).filter(Organization.slug == "test").first()
        if not test_org:
            # Create test organization
            test_org = Organization(
                name="Test Organization",
                slug="test",
                description="Test organization for automated tests",
            )
            db.add(test_org)
            db.flush()

            # Create test admin user
            test_admin = User(
                first_name="Test",
                last_name="Admin",
                email="test-admin@example.com",
                email_verified=True,
                password_hash=get_password_hash("testpassword"),
                is_admin=True,
            )
            db.add(test_admin)
            db.flush()  # Flush to get the user ID

            # Create user-organization relationship
            from models import UserOrganization

            user_org = UserOrganization(
                user_id=test_admin.id, organization_id=test_org.id, is_admin=True
            )
            db.add(user_org)
            db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error setting up test organization: {str(e)}")
    finally:
        db.close()

    return _client


@lru_cache
def _create_client() -> Client:
    test_settings = PlatformTestSettings()
    logger.info(test_settings)
    # If running as post deploy test, use the platform routes
    if test_settings.platform_routes:
        route = _convert_to_route(test_settings)
        route = route[:-1] if route.endswith("/") else route
        logger.info(f"Testing against url: {route}")
        client = Client(base_url=route)
    else:
        logger.info("Testing against url: http://testserver")
        client = TestClient(app=app)

    return client


def _convert_to_route(test_settings: PlatformTestSettings) -> str:
    routes_json = base64.b64decode(test_settings.platform_routes)
    routes_dict = json.loads(routes_json)
    return list(routes_dict)[1]


_client = _create_client()
