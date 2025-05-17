# FastAPI Template

## Architecture

This FastAPI Template is built with a modern, scalable architecture consisting of several key components:

- **FastAPI Application**: Core REST API server with CORS middleware
- **Service Layer**: Modular services for different domain functionalities (users, items, notifications, etc.)
- **Background Processing**: Service scheduler managing periodic tasks and Kafka consumers
- **Event Processing**: Kafka-based message broker for asynchronous operations
- **Database**: PostgreSQL for persistent storage, Redis for caching and session management
- **External Integrations**: Webhook support for third-party services
- **Authentication**: JWT-based authentication with refresh tokens

## Features

- **User Authentication**: Complete authentication system with JWT tokens and refresh tokens
- **Database Integration**: SQLAlchemy ORM with PostgreSQL
- **Caching**: Redis integration for caching
- **Background Tasks**: Scheduler for running periodic tasks
- **Asynchronous Processing**: Kafka integration for message processing
- **Notifications**: Notification system for sending messages to users
- **Webhooks**: Webhook system for integrating with external services
- **LLM Integration**: Anthropic Claude integration for AI capabilities
- **Docker Support**: Docker Compose setup for local development
- **Flexible Storage**: Support for both local file storage and Amazon S3 storage

## Developer Setup

### Development dependencies
- Python v3.11+
- Docker
- Docker Compose
- VS Code (recommended)
- Poetry v1.8.4

## Deployment
### Prod
The application can be deployed using Platform.sh or any other cloud provider that supports Python applications.

## Setup

### Quick Setup with Custom Project Name

When you clone this repository, you can use the provided setup script to customize the project name and avoid port conflicts:

```bash
# Basic usage (generates random ports)
./setup_project.py --name my_awesome_api

# Custom ports
./setup_project.py --name my_awesome_api --postgres-port 30100 --redis-port 30101 --pgadmin-port 16600 --kafka-port 9100

# Get help
./setup_project.py --help
```

This script will:
1. Replace all occurrences of "fastapitemplate" with your custom project name
2. Update port configurations to avoid conflicts
3. Update the API title and welcome message
4. Update database configuration in alembic migrations
5. Use unique environment variable prefixes to prevent conflicts between projects

### Manual Setup

Before you run the service, you will need to set up environment variables.

```bash
# Set environment variable to run in dev
echo "FASTAPI_ENV=dev" >> ~/.zshrc

# Setup python 3.11 environment
brew install poetry

# Install dependencies
poetry install

# Run the application
make app-run
```

## Run Application
This will run the application and spin up the needed docker containers
```shell
make app-run
```

## Run Docker Containers
```shell
make docker-up # spin up all docker containers and run the migration scripts

make docker-down # spin docker containers down
```

## Combined Commands
The following commands allow you to run multiple services together:

```shell
# Run API, Tasks, and UI services together (separate processes)
make run-all

# Run API with integrated tasks (tasks run within the API process)
make app-run-with-tasks

# Run API with integrated tasks and UI (fewer processes to manage)
make run-all-integrated
```

### Integrated Tasks
The API server can optionally run the task scheduler within the same process, which simplifies deployment and reduces the number of processes you need to manage. You can enable this feature in two ways:

1. Using the Makefile commands:
   ```shell
   make app-run-with-tasks
   # or
   make run-all-integrated
   ```

2. Directly with command-line arguments:
   ```shell
   python server.py --enable-tasks
   ```

This integration allows you to:
- Reduce the number of processes you need to manage
- Simplify deployment in production environments
- Ensure tasks and API are always in sync

### Testing
You can run tests using the following command:
```shell
make app-test
```

## Migration Commands
```shell
make migration-run # Runs the migrations against the db
make migration-new m="foo" # Creates a new migration script
```

## Project Structure

```
.
├── alembic/                  # Database migrations
├── consumers/                # Kafka consumers
│   ├── __init__.py
│   ├── kafka_config.py       # Kafka configuration
│   └── notification_consumer.py  # Notification consumer
├── services/                 # Service modules
│   ├── items.py              # Item service
│   ├── notifications.py      # Notification service
│   ├── storage.py            # Storage service (local and S3)
│   ├── users.py              # User service
│   └── webhooks.py           # Webhook service
├── static/                   # Static files
├── alembic.ini               # Alembic configuration
├── auth.py                   # Authentication utilities
├── db.py                     # Database configuration
├── docker-compose.yml        # Docker Compose configuration
├── env.py                    # Environment configuration
├── kafka_consumer.py         # Kafka consumer framework
├── kafka_utils.py            # Kafka utilities
├── llm.py                    # LLM integration
├── Makefile                  # Makefile for common commands
├── memcache.py               # Redis cache utilities
├── models.py                 # Database models
├── pyproject.toml            # Poetry configuration
├── README.md                 # This file
├── scheduler.py              # Background task scheduler
└── server.py                 # Main FastAPI application
```

## Extending the Template

### Adding a New Service

1. Create a new file in the `services/` directory
2. Define a FastAPI router and endpoints
3. Add the router to `server.py`

Example:

```python
# services/my_service.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import validate_jwt
from db import get_db

router = APIRouter()

@router.get("/")
async def get_my_data(db: Session = Depends(get_db), user: dict = Depends(validate_jwt)):
    # Implement your service logic here
    return {"message": "Hello from my service!"}
```

Then add to `server.py`:

```python
from services.my_service import router as my_service_router

# ...

app.include_router(my_service_router, prefix="/my-service", tags=["my-service"])
```

### Adding a New Kafka Consumer

1. Create a new file in the `consumers/` directory
2. Implement your consumer logic
3. Add the topic to `kafka_consumer.py`

### Adding a New Scheduled Task

1. Add a new method to the `ServiceScheduler` class in `scheduler.py`
2. Schedule the task in the `start` method

## Environment Variables

The application uses environment variables for configuration. You can set these in a `.env.dev` file for development.

> **Note:** After running the setup script, all environment variables will be prefixed with your project name in uppercase followed by an underscore (e.g., `MYPROJECT_SQL_HOST` instead of `SQL_HOST`). This prevents conflicts when running multiple projects on the same machine.

### Using the Environment Template (Local Development)

A `.env.template` file is provided as a reference for all required and optional environment variables. To set up your environment:

1. Copy the template to create your environment-specific files:
   ```bash
   cp .env.template .env.dev   # For development
   cp .env.template .env.prod  # For production
   ```

2. Edit the files to fill in your actual values for each environment.

3. Make sure not to commit your actual `.env.*` files to the repository (they are excluded in `.gitignore`).

#### Production

Use environment variable settings in your preferred cloud provider to keep secrets secure.

## S3 Storage Configuration

The application supports both local file storage and Amazon S3 storage. By default, it uses local storage, but you can enable S3 storage by setting the `USE_S3_STORAGE` environment variable to "true".

### Setting Up S3 Storage

1. Create S3 buckets in the desired regions:
   ```bash
   # Create bucket in US region
   aws s3api create-bucket --bucket ph-api-us --region us-east-1
   
   # Create bucket in India region
   aws s3api create-bucket --bucket ph-api-india --region ap-south-1 --create-bucket-configuration LocationConstraint=ap-south-1
   ```

2. Update your environment file (e.g., `.env.dev`) with the following settings:
   ```
   # Storage configuration
   HUB_API_USE_S3_STORAGE="true"
   
   # S3 credentials
   HUB_API_S3_ACCESS_KEY="your-aws-access-key"
   HUB_API_S3_SECRET_KEY="your-aws-secret-key"
   ```

3. Optionally, you can customize the bucket names and endpoints:
   ```
   HUB_API_S3_BUCKET_US="your-custom-us-bucket"
   HUB_API_S3_ENDPOINT_US="https://your-custom-endpoint.com"
   HUB_API_S3_BUCKET_INDIA="your-custom-india-bucket"
   HUB_API_S3_ENDPOINT_INDIA="https://your-custom-endpoint.com"
   ```

### How S3 Storage Works

The application uses the `boto3` library to interact with S3. When S3 storage is enabled:

1. Files are stored in S3 buckets instead of the local filesystem
2. The bucket is selected based on the project's data region (US or India)
3. Files are organized in the S3 bucket using the same path structure as local storage
4. Folders in S3 are represented by empty objects with trailing slashes

### Project Data Regions

Each project can be configured with a specific data region:

- `US`: Files are stored in the US bucket
- `INDIA`: Files are stored in the India bucket
- `LOCAL`: Files are stored locally, regardless of the global S3 setting

This allows you to control where data is stored on a per-project basis, which is useful for compliance with data residency requirements.
