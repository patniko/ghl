from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from loguru import logger
from auth import validate_jwt
from db import get_db
from models import (
    Organization,
    User,
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
    Check,
)
from utilities.initialize_checks import init_system_checks
from middleware import validate_admin_user

router = APIRouter()


@router.get("", response_model=List[OrganizationResponse])
async def get_organizations(
    db: Session = Depends(get_db),
):
    """
    Get all organizations that the user has access to.

    For regular users, this will return only their own organization.
    For admin users, this will return all organizations.
    """
    stmt = select(Organization).order_by(Organization.name)
    result = db.execute(stmt)
    organizations = result.scalars().all()
    return organizations


@router.get("/me", response_model=List[OrganizationResponse])
async def get_my_organizations(
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """
    Get all organizations that the user has access to.

    For regular users, this will return only organizations they belong to.
    For admin users, this will return all organizations.
    """
    from models import UserOrganization

    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # If the user is an admin, return all organizations
    if user.is_admin:
        stmt = select(Organization).order_by(Organization.name)
        result = db.execute(stmt)
        organizations = result.scalars().all()
        return organizations

    # Otherwise, return only the organizations the user belongs to
    user_orgs = (
        db.query(UserOrganization).filter(UserOrganization.user_id == user.id).all()
    )
    if not user_orgs:
        return []

    org_ids = [user_org.organization_id for user_org in user_orgs]
    organizations = db.query(Organization).filter(Organization.id.in_(org_ids)).all()

    return organizations


@router.get("/{slug}", response_model=OrganizationResponse)
async def get_organization_by_slug(
    slug: str,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get an organization by its slug"""
    from models import UserOrganization

    # Get the user
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get the requested organization
    org = db.query(Organization).filter(Organization.slug == slug).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Check if the user has access to this organization
    if not user.is_admin:
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org.id,
            )
            .first()
        )

        if not user_org:
            raise HTTPException(
                status_code=403, detail="You do not have access to this organization"
            )

    return org


@router.post("", response_model=OrganizationResponse)
async def create_organization(
    organization: OrganizationCreate,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """
    Create a new organization.

    Admin users can create organizations at any time.
    Regular users can create organizations only if they don't belong to any organization yet.
    """
    from models import UserOrganization

    # Check if the user exists
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # If the user is not an admin, check if they already belong to an organization
    if not user.is_admin:
        user_orgs = (
            db.query(UserOrganization).filter(UserOrganization.user_id == user.id).all()
        )
        if user_orgs:
            raise HTTPException(
                status_code=403,
                detail="You already belong to an organization. Only admin users can create additional organizations.",
            )

    # Create the organization
    try:
        new_org = Organization(
            name=organization.name, slug=organization.slug, description=None
        )
        db.add(new_org)
        db.commit()
        db.refresh(new_org)

        # Add the user to the organization
        from models import UserOrganization

        user_org = UserOrganization(
            user_id=user.id,
            organization_id=new_org.id,
            is_admin=True,  # Make the creator an admin of the organization
        )
        db.add(user_org)
        db.commit()

        # Initialize system checks for the new organization
        init_system_checks(new_org.id)

        return new_org
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Organization with slug '{organization.slug}' already exists",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error creating organization: {str(e)}"
        )


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: int,
    organization: OrganizationUpdate,
    organization_admin: Organization = Depends(validate_admin_user),
    db: Session = Depends(get_db),
):
    """
    Update an organization.

    Only admin users can update organizations.
    """
    # Get the organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Update the organization
    if organization.name is not None:
        org.name = organization.name
    if organization.description is not None:
        org.description = organization.description

    db.commit()
    db.refresh(org)
    return org


@router.delete("/{org_id}")
async def delete_organization(
    org_id: int,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """
    Delete an organization.

    Organization can be deleted by:
    - App admins
    - Organization admins
    """
    from models import UserOrganization

    # Get the user
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get the organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Check if user has permission to delete the organization
    if not user.is_admin:
        # Check if user is an admin of this organization
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .first()
        )
        if not user_org:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to delete this organization",
            )

    try:
        # Import all required models
        from models import (
            Batch,
            File,
            SyntheticDataset,
            DicomFile,
            Model,
            ColumnMapping,
            Check,
            UserOrganization,
        )

        # Delete all associated data in the correct order to handle foreign key constraints

        # 1. Delete user-organization relationships
        db.query(UserOrganization).filter(
            UserOrganization.organization_id == org_id
        ).delete(synchronize_session=False)

        # 2. Delete system checks
        db.query(Check).filter(Check.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 3. Delete files first (they have foreign key to batches)
        db.query(File).filter(File.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 4. Delete DICOM files
        db.query(DicomFile).filter(DicomFile.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 5. Delete batches
        db.query(Batch).filter(Batch.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 6. Delete synthetic datasets
        db.query(SyntheticDataset).filter(
            SyntheticDataset.organization_id == org_id
        ).delete(synchronize_session=False)

        # 7. Delete models
        db.query(Model).filter(Model.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 8. Delete column mappings
        db.query(ColumnMapping).filter(ColumnMapping.organization_id == org_id).delete(
            synchronize_session=False
        )

        # 9. Finally delete the organization
        db.delete(org)
        db.commit()
    except Exception as e:
        logger.error(f"Error deleting organization: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error deleting organization: {str(e)}"
        )

    return {"message": f"Organization '{org.name}' deleted successfully"}


@router.post("/{org_id}/reset-checks")
async def reset_system_checks(
    org_id: int,
    organization_admin: Organization = Depends(validate_admin_user),
    db: Session = Depends(get_db),
):
    """
    Reset system checks for an organization to their default state.
    This will delete all existing system checks and recreate them.
    """
    # Get the organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Delete existing system checks
    db.query(Check).filter(
        Check.organization_id == org_id,
        Check.is_system == True,  # noqa
    ).delete()
    db.commit()

    # Reinitialize system checks
    success = init_system_checks(org_id)
    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to reinitialize system checks"
        )

    return {
        "message": f"System checks reset successfully for organization '{org.name}'"
    }


@router.get("/{org_id}/users")
async def get_organization_users(
    org_id: int,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all users in an organization"""
    from models import UserOrganization

    # Check if the user has access to this organization
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get the organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Check if the user has access to this organization
    if not user.is_admin:
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org_id,
            )
            .first()
        )
        if not user_org:
            raise HTTPException(
                status_code=403, detail="You do not have access to this organization"
            )

    # Get all users in the organization
    user_orgs = (
        db.query(UserOrganization)
        .filter(UserOrganization.organization_id == org_id)
        .all()
    )

    user_ids = [user_org.user_id for user_org in user_orgs]
    users = db.query(User).filter(User.id.in_(user_ids)).all()

    # Include is_admin status for each user
    user_dict = {user_org.user_id: user_org.is_admin for user_org in user_orgs}

    return [
        {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_admin": user_dict[user.id],
        }
        for user in users
    ]


@router.post("/{org_id}/users")
async def add_user_to_organization(
    org_id: int,
    user_data: dict,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Add a user to an organization"""
    from models import UserOrganization

    # Check if the current user has admin access to this organization
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_admin:
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .first()
        )
        if not user_org:
            raise HTTPException(
                status_code=403,
                detail="You do not have admin access to this organization",
            )

    # Get the user to add
    user_to_add = db.query(User).filter(User.id == user_data["user_id"]).first()
    if not user_to_add:
        raise HTTPException(status_code=404, detail="User to add not found")

    # Check if the user is already in the organization
    existing_user_org = (
        db.query(UserOrganization)
        .filter(
            UserOrganization.user_id == user_to_add.id,
            UserOrganization.organization_id == org_id,
        )
        .first()
    )
    if existing_user_org:
        raise HTTPException(
            status_code=400, detail="User is already a member of this organization"
        )

    # Add the user to the organization
    user_org = UserOrganization(
        user_id=user_to_add.id,
        organization_id=org_id,
        is_admin=user_data.get("is_admin", False),
    )
    db.add(user_org)
    db.commit()

    return {"message": "User added to organization successfully"}


@router.delete("/{org_id}/users/{user_id}")
async def remove_user_from_organization(
    org_id: int,
    user_id: int,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Remove a user from an organization"""
    from models import UserOrganization

    # Check if the current user has admin access to this organization
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_admin:
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .first()
        )
        if not user_org:
            raise HTTPException(
                status_code=403,
                detail="You do not have admin access to this organization",
            )

    # Get the user to remove
    user_to_remove = (
        db.query(UserOrganization)
        .filter(
            UserOrganization.user_id == user_id,
            UserOrganization.organization_id == org_id,
        )
        .first()
    )
    if not user_to_remove:
        raise HTTPException(
            status_code=404, detail="User is not a member of this organization"
        )

    # Don't allow removing the last admin
    if user_to_remove.is_admin:
        admin_count = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .count()
        )
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove the last admin from the organization",
            )

    # Remove the user from the organization
    db.delete(user_to_remove)
    db.commit()

    return {"message": "User removed from organization successfully"}


@router.put("/{org_id}/users/{user_id}")
async def update_user_role(
    org_id: int,
    user_id: int,
    role_data: dict,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Update a user's role in an organization"""
    from models import UserOrganization

    # Check if the current user has admin access to this organization
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_admin:
        user_org = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.user_id == user.id,
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .first()
        )
        if not user_org:
            raise HTTPException(
                status_code=403,
                detail="You do not have admin access to this organization",
            )

    # Get the user to update
    user_to_update = (
        db.query(UserOrganization)
        .filter(
            UserOrganization.user_id == user_id,
            UserOrganization.organization_id == org_id,
        )
        .first()
    )
    if not user_to_update:
        raise HTTPException(
            status_code=404, detail="User is not a member of this organization"
        )

    # Don't allow removing admin status from the last admin
    if user_to_update.is_admin and not role_data.get("is_admin", True):
        admin_count = (
            db.query(UserOrganization)
            .filter(
                UserOrganization.organization_id == org_id,
                UserOrganization.is_admin == True,  # noqa
            )
            .count()
        )
        if admin_count <= 1:
            raise HTTPException(
                status_code=400, detail="Cannot remove admin status from the last admin"
            )

    # Update the user's role
    user_to_update.is_admin = role_data.get("is_admin", user_to_update.is_admin)
    db.commit()

    return {"message": "User role updated successfully"}


@router.get("/users/search")
async def search_users(
    q: str,
    current_user: dict = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Search for users by email or name"""
    # Search for users by email or name
    users = (
        db.query(User)
        .filter(
            (User.email.ilike(f"%{q}%"))
            | (User.first_name.ilike(f"%{q}%"))
            | (User.last_name.ilike(f"%{q}%"))
        )
        .limit(10)
        .all()
    )

    return [
        {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        }
        for user in users
    ]
