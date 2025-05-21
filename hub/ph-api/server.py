import os
import sys
import argparse
import threading
from contextlib import asynccontextmanager
import tomllib
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from middleware import add_large_file_upload_middleware
from tasks.scheduler import TaskScheduler

from services.organizations import router as organizations_router
from services.users import router as user_router
from services.webhooks import router as webhooks_router
from services.notifications import router as notifications_router
from services.batches import router as batches_router
from services.batch_processing import router as batch_processing_router
from services.files import router as files_router
from services.projects import router as projects_router
from services.samples import router as samples_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    run_scheduler_thread()

    yield
    # Shutdown (if needed)
    pass


app = FastAPI(
    title="HUB_API",
    description="A generic FastAPI template with authentication, database, and Kafka integration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add middleware for large file uploads (5GB limit)
add_large_file_upload_middleware(app, max_upload_size=5 * 1024 * 1024 * 1024)  # 5GB

# Mounts
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global variable to hold the task scheduler
task_scheduler = None

# Scheduler function
def run_scheduler_thread():
    """Start the task scheduler in a daemon thread if enabled"""
    global task_scheduler
    
    # Check if integrated tasks are enabled
    if os.environ.get("ENABLE_INTEGRATED_TASKS", "").lower() in ("true", "1", "yes"):
        from loguru import logger
        logger.info("Starting integrated task scheduler")
        
        # Get enabled tasks from environment configuration
        task_scheduler = TaskScheduler()
        scheduler_thread = threading.Thread(target=task_scheduler.start, daemon=True)
        scheduler_thread.start()
        logger.info("Integrated task scheduler started")
    else:
        # Tasks not enabled, do nothing
        pass


app.include_router(organizations_router, prefix="/orgs", tags=["organizations"])
app.include_router(
    organizations_router, prefix="/orgs/{org_slug}", tags=["organizations"]
)
app.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])
app.include_router(
    notifications_router, prefix="/notifications", tags=["notifications"]
)
# Users are independent and not organization-scoped
app.include_router(user_router, prefix="/users", tags=["users"])

app.include_router(batches_router, prefix="/batches/{org_slug}", tags=["batches"])
app.include_router(batch_processing_router, prefix="/batches/{org_slug}/processing", tags=["batch-processing"])
app.include_router(batch_processing_router, prefix="/batches/processing", tags=["batch-processing"])
app.include_router(files_router, prefix="/{org_slug}/files", tags=["files"])
app.include_router(projects_router, prefix="/projects/{org_slug}", tags=["projects"])
app.include_router(samples_router, prefix="/{org_slug}/samples", tags=["samples"])

# Load version and name from pyproject.toml
def get_project_metadata():
    """Read project metadata from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            return {
                "name": pyproject_data["tool"]["poetry"]["name"],
                "version": pyproject_data["tool"]["poetry"]["version"],
            }
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        # Fallback values if file can't be read or parsed
        return {"name": "nurture-api", "version": "0.1.0"}


@app.api_route("/", status_code=200, methods=["GET", "HEAD"])
async def load_root():
    """
    Root endpoint that returns basic API status information.
    Follows best practices for API health/status endpoints.
    """
    metadata = get_project_metadata()
    return {
        "status": "healthy",
        "version": metadata["version"],
        "name": metadata["name"],
        "environment": os.getenv("NURTURE_ENV", "dev"),
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GHL Progress Hub API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8080)),
        help="Port to run the server on",
    )
    parser.add_argument(
        "--enable-tasks",
        action="store_true",
        help="Enable integrated task scheduler",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the logging level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set environment variables based on arguments
    if args.enable_tasks:
        os.environ["ENABLE_INTEGRATED_TASKS"] = "true"
        print("Integrated tasks enabled")
    
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    
    # Configure Uvicorn with a larger request body size limit (5GB)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        limit_concurrency=10,  # Limit concurrent connections for large uploads
        timeout_keep_alive=300,  # Increase keep-alive timeout for large uploads
        # Note: Uvicorn doesn't directly expose a way to set the max request size
        # The middleware we added handles this at the FastAPI level
    )
