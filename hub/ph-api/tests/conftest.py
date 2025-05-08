import pytest
from db import get_db
from tests.test_utils import setup_test_organization, teardown_test_organization


@pytest.fixture
def test_organization():
    """
    Pytest fixture for setting up a test organization.

    Usage:
        def test_something(test_organization):
            org, admin, db = test_organization
            # Use org and admin in your test

    Returns:
        Tuple of (organization, admin_user, db_session)
    """
    db = next(get_db())
    org, admin = setup_test_organization(db)

    yield org, admin, db

    # Cleanup
    teardown_test_organization(db, org.slug)
