import os
from functools import lru_cache

from loguru import logger
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

APP_ENV = os.environ.get("FASTAPI_ENV", "prod")
logger.info(f"Running with FASTAPI_ENV: {APP_ENV}")


# Determine the project name from environment variables
# Look for any environment variable with _SQL_DATABASE suffix
def get_project_prefix():
    for key in os.environ:
        if key.endswith("_SQL_DATABASE"):
            return key.replace("_SQL_DATABASE", "_")
    return "HUB_API_"  # Default prefix if not found


PROJECT_PREFIX = get_project_prefix()
logger.info(f"Using project prefix: {PROJECT_PREFIX}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.base", f".env.{APP_ENV}"),
        env_prefix=PROJECT_PREFIX,
        extra="ignore",
    )

    auth_secret_key: str | None = None
    auth_algorithm: str | None = None
    auth_access_token_expire_minutes: int | None = None

    sql_host: str
    sql_port: str
    sql_user: str
    sql_password: str
    sql_database: str

    redis_host: str
    redis_port: str
    redis_password: str | None = None
    redis_db: int = 0

    kafka_host: str | None = None
    kafka_port: str | None = None

    def kafka_server(self):
        return self.kafka_host + ":" + self.kafka_port

    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_image_upload_url: str | None = None

    twilio_client_id: str | None = None
    twilio_client_key: str | None = None
    twilio_verify: str | None = None

    apns_key_id: str | None = None
    apns_team_id: str | None = None
    apns_bundle_id: str | None = None
    is_production: bool = False

    # Anthropic settings
    anthropic_api_key: str | None = None

    # Google Maps settings
    google_maps_api_key: str | None = None

    # Google OAuth settings
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_pubsub_topic: str | None = None

    # S3 storage settings
    use_s3_storage: bool = False
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket_us: str = "ph-api-us"
    s3_endpoint_us: str = "https://s3.us-east-1.amazonaws.com"
    s3_bucket_india: str = "ph-api-india"
    s3_endpoint_india: str = "https://s3.ap-south-1.amazonaws.com"

    app_env: str = APP_ENV


@lru_cache()
def get_settings():
    return Settings()


def get_sqlConnectionString():
    settings = get_settings()
    connection = (
        "postgresql://"
        + settings.sql_user
        + ":"
        + settings.sql_password
        + "@"
        + settings.sql_host
        + ":"
        + settings.sql_port
        + "/"
        + settings.sql_database
    )
    logger.debug(f"Connection string: {connection}")
    return connection
