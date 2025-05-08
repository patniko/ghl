"""
Middleware package for the FastAPI application.

This package contains middleware components that can be added to the FastAPI application
to modify its behavior, such as handling large file uploads, authentication, etc.
"""

# Import file upload middleware
from middleware.file_upload import (
    add_large_file_upload_middleware,
    LargeFileUploadMiddleware,
)

# Import authentication middleware
from middleware.auth import (
    get_organization_from_path,
    validate_user_organization,
    get_project_from_path,
    validate_admin_user,
)

__all__ = [
    # File upload middleware
    "add_large_file_upload_middleware",
    "LargeFileUploadMiddleware",
    # Authentication middleware
    "get_organization_from_path",
    "validate_user_organization",
    "get_project_from_path",
    "validate_admin_user",
]
