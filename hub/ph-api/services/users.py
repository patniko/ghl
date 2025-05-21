import io
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from models import OrganizationResponse
from uuid import uuid4

import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from twilio.rest import Client

from auth import create_access_token
from auth import get_user_by_id
from auth import validate_jwt
from auth import verify_password  # Add this import
from auth import get_password_hash  # Add this import
from auth import (
    create_refresh_token,
    validate_refresh_token,
    revoke_refresh_token,
    revoke_all_user_tokens,
)
from db import get_db
from env import get_settings
from memcache import get_cached_data
from memcache import set_cached_data
from models import User
from auth import get_user_organizations

router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user_id: int
    expires_in: int  # Access token expiration in seconds


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RevokeTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    email_verified: bool
    picture: str | None
    is_onboarded: bool
    organizations: List[OrganizationResponse] = []


class UserUpdate(BaseModel):
    first_name: Optional[str] | None = None
    last_name: Optional[str] | None = None
    picture: Optional[str] | None = None


class Token(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    email_verified: str
    picture: str
    updated_at: str
    admin: str
    access_token: str
    token_type: str


class HashResponse(BaseModel):
    key_name: str
    value: str

    model_config = {"from_attributes": True}


class HashCreate(BaseModel):
    key_name: str
    value: str


class UserHashes(BaseModel):
    contacts_hash: Optional[str]
    routines_hash: Optional[str]
    catalog_hash: Optional[str]


class ResetToken(BaseModel):
    confirmation_code: str
    new_password: str
    confirm_new_password: str


class PasswordUpdate(BaseModel):
    current_password: Optional[str] = None  # Optional for users without password
    new_password: str


# Service helper functions
def is_valid_phone_number(phone: str):
    if phone == "invalid_phone":
        return False
    else:
        return True


def sanitize_number(phone: str):
    # Remove any non-digit characters
    digits_only = "".join(filter(str.isdigit, phone))
    # If number started with a + we assume it is already a full number
    if phone[0] == "+":
        return f"+{digits_only}"
    # If it's a US number without country code, add +1
    if len(digits_only) == 10:
        return f"+1{digits_only}"
    # If it already has country code (11 digits starting with 1)
    elif len(digits_only) == 11 and digits_only.startswith("1"):
        return f"+{digits_only}"
    # Return with + prefix if not already present
    return f"+{digits_only}" if not phone.startswith("+") else phone


def ignore_validation(phone: str):
    return phone in [
        "+15625555555",
        "+15625551111",
        "+11234567890",
        "+15551234567",
        "+15625944162",
    ]


def resize_image(image: Image.Image, max_size: tuple) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size)
    return image


def update_user_avatar(user: User, picture: str, db: Session):
    if not user:
        raise HTTPException(status_code=404, detail="No user found.")
    try:
        if picture:
            user.picture = picture
        db.commit()

        user = get_user_by_id(db, user_id=user.id)
        # Get user organizations
        organizations = get_user_organizations(db, user.id)
        org_responses = [
            OrganizationResponse(
                id=org.id,
                name=org.name,
                slug=org.slug,
                description=org.description,
                created_at=org.created_at,
                updated_at=org.updated_at,
            )
            for org in organizations
        ]

        response: UserResponse = UserResponse(
            id=user.id,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            email_verified=user.email_verified,
            picture=user.picture,
            is_onboarded=True if user.first_name else False,
            organizations=org_responses,
        )

        # Update the avatar cache with new data
        cache_key = f"avatar:{user.id}"
        cache_data = {
            "id": user.id,
            "first_name": user.first_name,
            "last_initial": user.last_name[0] if user.last_name else "",
            "picture": user.picture,
        }
        try:
            set_cached_data(cache_key, cache_data, 3600)
        except Exception as e:
            logger.error(f"Error updating avatar cache for user {user.id}: {str(e)}")
            # Continue even if cache update fails
            pass

        return response
    except Exception as e:
        logger.error(f"Error updating user avatar: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update user avatar")


def create_token_response(
    user: User, db: Session, device_info: str = None
) -> TokenResponse:
    """Helper function to create a consistent token response"""
    # Create access token (24 hours)
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
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
            "password_set": bool(user.password_hash),
            "is_admin": user.is_admin,
        },
        expires_delta=access_token_expires,
    )

    # Create refresh token (90 days)
    refresh_token = create_refresh_token(db, user.id, device_info)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user.id,
        "expires_in": int(access_token_expires.total_seconds()),
    }


# Custom clients and settings
# twilio_client = Client(
#     get_settings().twilio_client_id, get_settings().twilio_client_key
# )
# twilio_verify = twilio_client.verify.services(get_settings().twilio_verify)

MAX_AVATAR_FILE_SIZE = 1 * 1024 * 1024  # 1MB
MAX_AVATAR_IMAGE_SIZE = (300, 300)


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    db: Session = Depends(get_db), user: dict = Depends(validate_jwt)
):
    try:
        user = get_user_by_id(db, user_id=user["user_id"])
        if not user:
            logger.warning(f"User not found for id: {user['user_id']}")
            raise HTTPException(status_code=404, detail="No user found")

        # Get user organizations
        organizations = get_user_organizations(db, user.id)
        org_responses = [
            OrganizationResponse(
                id=org.id,
                name=org.name,
                slug=org.slug,
                description=org.description,
                created_at=org.created_at,
                updated_at=org.updated_at,
            )
            for org in organizations
        ]

        return UserResponse(
            id=user.id,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            email_verified=user.email_verified,
            picture=user.picture,
            is_onboarded=True if user.first_name else False,
            organizations=org_responses,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in read_users_me: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/me", response_model=UserResponse)
async def update_users_me(
    update_data: dict,
    db: Session = Depends(get_db),
    user: dict = Depends(validate_jwt),
):
    try:
        db_user = get_user_by_id(db, user_id=user["user_id"])
        if not db_user:
            logger.warning("User not found for update")
            raise HTTPException(status_code=404, detail="No user found")

        try:
            # Parse the update data
            try:
                updates = UserUpdate(**update_data)
            except Exception as e:
                logger.error(f"Error parsing user update data: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid update data: {str(e)}"
                )

            if updates.first_name is not None:
                db_user.first_name = updates.first_name
            if updates.last_name is not None:
                db_user.last_name = updates.last_name
            if updates.picture is not None:
                db_user.picture = updates.picture or ""  # Convert None to empty string

            db.commit()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error updating user: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to update user")

        updated_user = db_user

        # Get user organizations
        organizations = get_user_organizations(db, updated_user.id)
        org_responses = [
            OrganizationResponse(
                id=org.id,
                name=org.name,
                slug=org.slug,
                description=org.description,
                created_at=org.created_at,
                updated_at=org.updated_at,
            )
            for org in organizations
        ]

        return UserResponse(
            id=updated_user.id,
            first_name=updated_user.first_name,
            last_name=updated_user.last_name,
            email=updated_user.email,
            email_verified=updated_user.email_verified,
            picture=updated_user.picture,
            is_onboarded=True if updated_user.first_name else False,
            organizations=org_responses,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_users_me: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/me/token", response_model=TokenResponse)
async def get_token(db: Session = Depends(get_db), user: dict = Depends(validate_jwt)):
    try:
        user = get_user_by_id(db, user_id=user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="No user found")

        if user.is_deleted:
            raise HTTPException(status_code=403, detail="Account has been deleted")

        # Get device info from request if available
        device_info = "Unknown device"  # Default value

        return create_token_response(user, db, device_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating tokens: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate tokens")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh an access token using a valid refresh token"""
    try:
        # Validate the refresh token
        token_data = validate_refresh_token(db, request.refresh_token)

        # Get the user
        user = get_user_by_id(db, user_id=token_data["user_id"])
        if not user:
            logger.warning("User not found for refresh token")
            raise HTTPException(status_code=404, detail="User not found")

        if user.is_deleted:
            # Revoke the token if user is deleted
            revoke_refresh_token(db, request.refresh_token)
            raise HTTPException(status_code=403, detail="Account has been deleted")

        # Create a new access token with the same refresh token
        access_token_expires = timedelta(hours=24)
        access_token = create_access_token(
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
                "password_set": bool(user.password_hash),
                "is_admin": user.is_admin,
            },
            expires_delta=access_token_expires,
        )

        return {
            "access_token": access_token,
            "refresh_token": request.refresh_token,  # Return the same refresh token
            "token_type": "bearer",
            "user_id": user.id,
            "expires_in": int(access_token_expires.total_seconds()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")


@router.post("/revoke")
async def revoke_token(
    request: RevokeTokenRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(validate_jwt),
):
    """Revoke a refresh token"""
    try:
        # Revoke the token
        success = revoke_refresh_token(db, request.refresh_token)
        if not success:
            raise HTTPException(status_code=404, detail="Token not found")

        return {"message": "Token revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to revoke token")


@router.post("/revoke-all")
async def revoke_all_tokens(
    db: Session = Depends(get_db), user: dict = Depends(validate_jwt)
):
    """Revoke all refresh tokens for the current user"""
    try:
        # Revoke all tokens for the user
        success = revoke_all_user_tokens(db, user["user_id"])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to revoke all tokens")

        return {"message": "All tokens revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking all tokens: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to revoke all tokens")


@router.post("/me/avatar", response_model=UserResponse)
async def upload_image(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: dict = Depends(validate_jwt),
):
    # Upload user avatar with proper error handling
    try:
        db_user = get_user_by_id(db, user_id=user["user_id"])
        if not db_user:
            logger.warning("User not found for avatar upload")
            raise HTTPException(status_code=404, detail="No user found")

        # Validate file size
        try:
            await image.seek(0)
            content = await image.read()
            if len(content) > MAX_AVATAR_FILE_SIZE:
                logger.warning(
                    f"Avatar upload exceeds size limit: {len(content)} bytes"
                )
                raise HTTPException(
                    status_code=400, detail="File size exceeds 1MB limit"
                )
        except Exception as e:
            logger.error(f"Error reading upload file: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid file upload")

        # Process image
        try:
            img = Image.open(io.BytesIO(content))
            img = resize_image(img, MAX_AVATAR_IMAGE_SIZE)

            buf = io.BytesIO()
            img.save(buf, format=img.format)
            buf.seek(0)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Upload to Cloudflare
        try:
            headers = {"Authorization": f"Bearer {get_settings().cloudflare_api_token}"}

            files = {"file": (image.filename, buf, image.content_type)}

            unique_identifier = f"user_avatar_{db_user.id}"
            params = {"id": unique_identifier}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_settings().cloudflare_image_upload_url,
                    headers=headers,
                    files=files,
                    params=params,
                )

                if response.status_code != 200:
                    logger.error(f"Cloudflare upload failed: {response.text}")
                    raise HTTPException(
                        status_code=500, detail="Failed to upload image"
                    )

                payload = response.json()
                picture = payload["result"]["variants"][0]
                return update_user_avatar(db_user, picture, db)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during image upload: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload image")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/avatar/{user_id}")
async def get_avatar(user_id: int, db: Session = Depends(get_db)):
    """Get user avatar with Redis caching and graceful fallback"""
    cache_key = f"avatar:{user_id}"

    try:
        # Try to get from cache first
        cached_data = get_cached_data(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for avatar:{user_id}")
            return cached_data
    except Exception as e:
        # Log cache error but continue to database
        logger.error(f"Cache error for avatar:{user_id}: {str(e)}")

    # Get from database (either cache miss or cache error)
    try:
        user = get_user_by_id(db, user_id=user_id)
        if not user:
            logger.error(f"No user found for avatar:{user_id}")
            raise HTTPException(status_code=404, detail="No user found.")

        # Prepare response data
        response_data = {
            "id": user_id,
            "first_name": user.first_name,
            "last_initial": user.last_name[0] if user.last_name else "",
            "picture": user.picture,
        }
        # Try to store in cache, but don't block on cache errors
        try:
            cache_success = set_cached_data(cache_key, response_data, 3600)
            if not cache_success:
                logger.warning(f"Failed to cache avatar:{user_id}")
        except Exception as e:
            logger.error(f"Error caching avatar:{user_id}: {str(e)}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error for avatar:{user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/me")
async def delete_account(
    db: Session = Depends(get_db), user: dict = Depends(validate_jwt)
):
    """Soft delete a user account"""
    try:
        db_user = get_user_by_id(db, user_id=user["user_id"])
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Mark user as deleted
        db_user.is_deleted = True
        db_user.deleted_at = datetime.now(timezone.utc)
        # Update email to prevent reuse with random suffix
        random_suffix = str(uuid4().int % 10000)  # Get random number between 0-9999
        email_parts = db_user.email.split("@")
        db_user.email = f"{email_parts[0]}__DELETED__{random_suffix}@{email_parts[1]}"

        # Revoke all refresh tokens
        revoke_all_user_tokens(db, db_user.id)

        db.commit()
        return {"message": "Account successfully deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting account: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete account")


@router.post("/password/set", response_model=TokenResponse)
async def set_password(
    password_update: PasswordUpdate,
    db: Session = Depends(get_db),
    user: dict = Depends(validate_jwt),
):
    """Set or update user password"""
    try:
        db_user = get_user_by_id(db, user_id=user["user_id"])
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update password
        db_user.password_hash = get_password_hash(password_update.new_password)
        db.commit()

        # Generate new tokens
        device_info = "Password set device"
        return create_token_response(db_user, db, device_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting password: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to set password")


class UserRegister(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str


class PasswordLogin(BaseModel):
    email: str
    password: str


@router.post("/register")
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user with password"""
    try:
        # Check if user already exists with this email
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            if existing_user.is_deleted:
                raise HTTPException(status_code=403, detail="Account has been deleted")
            raise HTTPException(
                status_code=400, detail="User with this email already exists"
            )

        # Create the user without organization
        new_user = User(
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            email=user_data.email,
            email_verified=False,  # Email verification would be a separate process
            picture="",
            password_hash=get_password_hash(user_data.password),
            last_logged_in=datetime.utcnow(),
            created_at=datetime.utcnow(),
            is_admin=False,
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {"message": "User registered successfully", "user_id": new_user.id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to register user")


@router.post("/password/login", response_model=TokenResponse)
async def login_with_password(
    password_login: PasswordLogin, db: Session = Depends(get_db)
):
    """Login using email and password"""
    try:
        user = db.query(User).filter(User.email == password_login.email).first()

        # First check if user exists at all
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Then check password matches if user has password login enabled
        if user.password_hash:
            if not verify_password(password_login.password, user.password_hash):
                raise HTTPException(status_code=401, detail="Invalid credentials")
        else:
            # User exists but hasn't set up password login
            raise HTTPException(
                status_code=400, detail="Password login not enabled for this user"
            )

        # Generate tokens
        device_info = "Password login device"
        return create_token_response(user, db, device_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during password login: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to login with password")
