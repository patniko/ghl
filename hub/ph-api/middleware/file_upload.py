from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp


class LargeFileUploadMiddleware(BaseHTTPMiddleware):
    """
    Middleware to increase the maximum file upload size for FastAPI.

    By default, FastAPI uses a 100MB limit for file uploads.
    This middleware increases that limit to allow for larger file uploads.
    """

    def __init__(
        self, app: ASGIApp, max_upload_size: int = 5 * 1024 * 1024 * 1024
    ):  # 5GB default
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        # Modify the body size limit for this request
        # This is a workaround as FastAPI doesn't expose a direct way to modify this limit
        request.scope["max_body_size"] = self.max_upload_size

        # Continue processing the request
        response = await call_next(request)
        return response


def add_large_file_upload_middleware(
    app: FastAPI, max_upload_size: int = 5 * 1024 * 1024 * 1024
):
    """
    Add middleware to increase the maximum file upload size.

    Args:
        app: The FastAPI application
        max_upload_size: Maximum upload size in bytes (default: 5GB)
    """
    app.add_middleware(LargeFileUploadMiddleware, max_upload_size=max_upload_size)
