from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from db import get_db
from models import Organization, User, UserOrganization, Project
from auth import validate_jwt


async def get_organization_from_path(request: Request) -> Organization:
    """
    Extract the organization slug from the request path and get the organization.

    The organization slug can be in different positions depending on the route:
    - For routes like "/orgs/{org_slug}/...", it's the second path segment
    - For routes like "/batches/{org_slug}/...", it's the second path segment
    - For routes like "/{org_slug}/files/...", it's the first path segment
    """
    path = request.url.path
    path_parts = path.strip("/").split("/")

    if not path_parts:
        raise HTTPException(
            status_code=404, detail="Organization slug not found in path"
        )

    # Determine the position of the organization slug based on the path
    if (
        path_parts[0] == "batches"
        or path_parts[0] == "datasets"
        or path_parts[0] == "dicom"
        or path_parts[0] == "data-quality"
        or path_parts[0] == "projects"
    ):
        # For routes like "/batches/{org_slug}/...", the slug is the second segment
        if len(path_parts) < 2:
            raise HTTPException(
                status_code=404, detail="Organization slug not found in path"
            )
        org_slug = path_parts[1]
    elif path_parts[0] == "orgs":
        # For routes like "/orgs/{org_slug}/...", the slug is the second segment
        if len(path_parts) < 2:
            raise HTTPException(
                status_code=404, detail="Organization slug not found in path"
            )
        org_slug = path_parts[1]
    else:
        # For routes like "/{org_slug}/files/...", the slug is the first segment
        org_slug = path_parts[0]

    # Get the organization from the database
    db = next(get_db())
    try:
        org = db.query(Organization).filter(Organization.slug == org_slug).first()
        if not org:
            raise HTTPException(
                status_code=404, detail=f"Organization '{org_slug}' not found"
            )

        return org
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving organization: {str(e)}"
        )


async def validate_user_organization(
    request: Request,
    organization: Organization = Depends(get_organization_from_path),
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
) -> Organization:
    """
    Validate that the authenticated user belongs to the organization.

    This dependency can be used in route handlers to ensure that the user
    has access to the organization's resources.
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

    # Get the user from the database
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if the user belongs to the organization using the many-to-many relationship
    user_org = (
        db.query(UserOrganization)
        .filter(
            UserOrganization.user_id == user_id,
            UserOrganization.organization_id == organization.id,
        )
        .first()
    )

    if not user_org:
        raise HTTPException(
            status_code=403, detail="You do not have access to this organization"
        )

    return organization


async def get_project_from_path(
    request: Request,
    organization: Organization = Depends(validate_user_organization),
    db: Session = Depends(get_db),
) -> Project:
    """
    Extract the project name from the request path and get the project.

    The project name is expected to be the second path segment.
    For example, in the path "/acme/project-x/...", "project-x" is the project name.
    """
    path = request.url.path
    path_parts = path.strip("/").split("/")

    if len(path_parts) < 2:
        raise HTTPException(status_code=404, detail="Project name not found in path")

    # The project name is the second path segment
    project_name = path_parts[1]

    # Get the project from the database
    project = (
        db.query(Project)
        .filter(
            Project.name == project_name, Project.organization_id == organization.id
        )
        .first()
    )

    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project '{project_name}' not found"
        )

    return project


def validate_admin_user(
    organization: Organization = Depends(validate_user_organization),
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
) -> Organization:
    """
    Validate that the authenticated user is an admin of the organization.

    This dependency can be used in route handlers that require admin privileges.
    """
    user_id = current_user.get("user_id")

    # Get the user-organization relationship from the database
    user_org = (
        db.query(UserOrganization)
        .filter(
            UserOrganization.user_id == user_id,
            UserOrganization.organization_id == organization.id,
        )
        .first()
    )

    if not user_org:
        raise HTTPException(
            status_code=404, detail="User-organization relationship not found"
        )

    # Check if the user is an admin of this organization
    if not user_org.is_admin:
        raise HTTPException(
            status_code=403, detail="Admin privileges required for this operation"
        )

    return organization
