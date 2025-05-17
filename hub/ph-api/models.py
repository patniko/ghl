import enum
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, validator
from sqlalchemy import ARRAY, Boolean, Column, Date, DateTime, Enum, ForeignKey
from sqlalchemy import Integer, String, Text, JSON, UniqueConstraint, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from db import Base

###
# Organizations and Users (Many-to-Many)
###


# Association table for User-Organization many-to-many relationship
class UserOrganization(Base):
    __tablename__ = "user_organizations"
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    organization_id = Column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    is_admin = Column(
        Boolean, default=False
    )  # Flag to mark user as admin of this organization
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="user_organizations")
    organization = relationship("Organization", back_populates="user_organizations")


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user_organizations = relationship(
        "UserOrganization", back_populates="organization", cascade="all, delete-orphan"
    )
    users = relationship("User", secondary="user_organizations", viewonly=True)
    batches = relationship("Batch", back_populates="organization")
    models = relationship("Model", back_populates="organization")
    checks = relationship("Check", back_populates="organization")
    column_mappings = relationship("ColumnMapping", back_populates="organization")
    files = relationship("File", back_populates="organization")
    dicom_files = relationship("DicomFile", back_populates="organization")
    synthetic_datasets = relationship("SyntheticDataset", back_populates="organization")
    projects = relationship("Project", back_populates="organization")


###
# Users
###


# Alchemy models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    email_verified = Column(Boolean, default=False)
    picture = Column(String, unique=False, nullable=True)
    password_hash = Column(String, nullable=True)
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_logged_in = Column(DateTime(timezone=True), nullable=True)
    timezone = Column(String, nullable=False, default="UTC")
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # Soft delete timestamp
    is_deleted = Column(Boolean, default=False)  # Flag to mark user as deleted
    is_admin = Column(
        Boolean, default=False
    )  # Flag to mark user as admin of the organization

    # Relationships
    user_organizations = relationship(
        "UserOrganization", back_populates="user", cascade="all, delete-orphan"
    )
    organizations = relationship(
        "Organization", secondary="user_organizations", viewonly=True
    )


class DeviceToken(Base):
    __tablename__ = "device_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token = Column(String, nullable=False, unique=True, index=True)
    device_type = Column(String, nullable=False)  # "ios" or "android"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token = Column(String, nullable=False, unique=True, index=True)
    device_info = Column(String, nullable=True)  # Store device fingerprint info
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


###
# Notifications
###


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    users = Column(ARRAY(Integer), nullable=False)
    event = Column(String, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sent_status = Column(Boolean, default=False)


###
# OAuth2 Credentials
###


class OAuth2Provider(str, enum.Enum):
    GOOGLE = "google"
    GITHUB = "github"
    FACEBOOK = "facebook"


class OAuth2Credentials(Base):
    __tablename__ = "oauth2_credentials"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider = Column(Enum(OAuth2Provider), nullable=False)
    email = Column(String, nullable=False, index=True)
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    token_expiry = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


###
# Generic Item (Example Model)
###


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    data = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization")


###
# Batches and Datasets
###


class Batch(Base):
    __tablename__ = "batches"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    processing_status = Column(String, nullable=True)  # Batch processing status
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    files = relationship("File", back_populates="batch")
    quality_summary = Column(JSON, nullable=True)  # Aggregated quality metrics

    # Relationships
    organization = relationship("Organization", back_populates="batches")
    project = relationship("Project", back_populates="batches")


class DataRegion(str, enum.Enum):
    """Data region options for project storage"""

    LOCAL = "local"  # Local storage (default)
    INDIA = "india"  # India data center
    US = "us"  # United States data center


class Project(Base):
    """Project model for organizing files and data with location settings"""

    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    data_region = Column(String, nullable=False, default=DataRegion.LOCAL.value)
    s3_bucket_name = Column(String, nullable=True)  # Custom bucket name if provided
    first_batch_id = Column(
        Integer, nullable=True
    )  # ID of the first batch created for this project
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Add unique constraint for name + organization_id
    __table_args__ = (
        # Ensure project names are unique within an organization
        # This prevents users from creating multiple projects with the same name in an organization
        # But allows different organizations to have projects with the same name
        UniqueConstraint("organization_id", "name", name="uix_project_org_name"),
    )

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    files = relationship("File", back_populates="project")
    batches = relationship("Batch", back_populates="project")


class SyntheticDataset(Base):
    __tablename__ = "synthetic_datasets"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    batch_id = Column(
        Integer, ForeignKey("batches.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    num_patients = Column(Integer, nullable=False)
    data = Column(JSON, nullable=True)  # Store the actual dataset
    column_mappings = Column(JSON, nullable=True)  # Maps columns to check types
    applied_checks = Column(JSON, nullable=True)  # Stores which checks were applied
    check_results = Column(JSON, nullable=True)  # Results of applied checks
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization", back_populates="synthetic_datasets")


###
# Check Catalog
###


# Data type constants (using strings instead of enums for database compatibility)
class DataType(str, enum.Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    TEXT = "text"
    BOOLEAN = "boolean"


# Check severity constants (using strings instead of enums for database compatibility)
class CheckSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Check scope constants (using strings instead of enums for database compatibility)
class CheckScope(str, enum.Enum):
    FIELD = "field"
    COLUMN = "column"
    ROW = "row"
    FILE = "file"
    BATCH = "batch"


class Check(Base):
    __tablename__ = "checks"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    data_type = Column(String, nullable=False)  # Using string instead of enum
    scope = Column(
        String, nullable=False, default=CheckScope.FIELD.value
    )  # Default to field-level
    parameters = Column(JSON, nullable=True)  # Default parameters for the check
    implementation = Column(String, nullable=False)  # Function name to execute
    is_system = Column(Boolean, default=False)  # System-provided vs user-created
    python_script = Column(
        Text, nullable=True
    )  # Python script for row/file level checks
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization", back_populates="checks")


class ColumnMapping(Base):
    __tablename__ = "column_mappings"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    column_name = Column(String, nullable=False, index=True)
    data_type = Column(String, nullable=False)  # Using string instead of enum
    description = Column(Text, nullable=True)
    synonyms = Column(ARRAY(String), nullable=True)  # Alternative names for matching
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization", back_populates="column_mappings")


###
# Models Catalog
###


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String, nullable=False)
    parameters = Column(JSON, nullable=True)  # Default parameters for the model
    implementation = Column(String, nullable=False)  # Implementation type
    is_system = Column(Boolean, default=False)  # System-provided vs user-created
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization", back_populates="models")


###
# Files
###


# Processing status constants
class ProcessingStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Batch processing status constants
class BatchProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# File type constants
class FileType:
    DICOM = "dicom"
    CSV = "csv"
    MP4 = "mp4"
    NPZ = "npz"
    JSON = "json"


class ECGAnalysis(Base):
    """Table for storing ECG analysis results with flat structure for analytics"""

    __tablename__ = "ecg_analyses"
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Signal quality metrics
    has_missing_leads = Column(Boolean, nullable=True)
    signal_noise_ratio = Column(Float, nullable=True)
    baseline_wander_score = Column(Float, nullable=True)
    motion_artifact_score = Column(Float, nullable=True)

    # R-R interval metrics
    rr_interval_mean = Column(Float, nullable=True)
    rr_interval_stddev = Column(Float, nullable=True)
    rr_interval_consistency = Column(Float, nullable=True)

    # QRS detection
    qrs_count = Column(Integer, nullable=True)
    qrs_detection_confidence = Column(Float, nullable=True)

    # HRV metrics
    hrv_sdnn = Column(Float, nullable=True)  # Standard deviation of NN intervals
    hrv_rmssd = Column(
        Float, nullable=True
    )  # Root mean square of successive differences
    hrv_pnn50 = Column(
        Float, nullable=True
    )  # Proportion of NN50 divided by total number of NNs
    hrv_lf = Column(Float, nullable=True)  # Low-frequency power
    hrv_hf = Column(Float, nullable=True)  # High-frequency power
    hrv_lf_hf_ratio = Column(Float, nullable=True)  # LF/HF ratio

    # Frequency content analysis
    frequency_peak = Column(Float, nullable=True)
    frequency_power_vlf = Column(Float, nullable=True)  # Very low frequency power
    frequency_power_lf = Column(Float, nullable=True)  # Low frequency power
    frequency_power_hf = Column(Float, nullable=True)  # High frequency power

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationship
    file = relationship("File", back_populates="ecg_analysis")


class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    batch_id = Column(
        Integer,
        ForeignKey("batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    content_type = Column(String, nullable=False)
    file_type = Column(String, nullable=False, index=True)  # dicom, csv, mp4, npz

    # Metadata stored as JSON
    file_metadata = Column(JSON, nullable=True)

    # For CSV files, store headers
    csv_headers = Column(JSON, nullable=True)

    # For CSV files, store potential column-to-check mappings
    potential_mappings = Column(JSON, nullable=True)

    # Schema type for identifying specific file formats (e.g., "alivecor" for AliveCor ECG JSON files)
    schema_type = Column(String, nullable=True)

    # Processing status and results
    processing_status = Column(String, nullable=False, default=ProcessingStatus.PENDING)
    processing_results = Column(JSON, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Thumbnail information
    thumbnail = Column(String, nullable=True)  # Path or base64 encoded thumbnail
    has_thumbnail = Column(
        Boolean, default=False
    )  # Flag to indicate if thumbnail was generated
    thumbnail_generated_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    batch = relationship("Batch", back_populates="files")
    organization = relationship("Organization", back_populates="files")
    project = relationship("Project", back_populates="files")
    ecg_analysis = relationship(
        "ECGAnalysis",
        back_populates="file",
        uselist=False,
        cascade="all, delete-orphan",
    )


###
# DICOM Files
###


class DicomFile(Base):
    __tablename__ = "dicom_files"
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    content_type = Column(String, nullable=False)

    # DICOM metadata
    patient_id = Column(String, nullable=True, index=True)
    patient_name = Column(String, nullable=True)
    batch_instance_uid = Column(String, nullable=True, index=True)
    series_instance_uid = Column(String, nullable=True, index=True)
    sop_instance_uid = Column(String, nullable=True, index=True)
    modality = Column(String, nullable=True)  # CT, MRI, etc.
    batch_date = Column(Date, nullable=True)

    # Additional metadata stored as JSON
    dicom_metadata = Column(JSON, nullable=True)

    # Processing status and results
    processing_status = Column(String, nullable=False, default=ProcessingStatus.PENDING)
    processing_results = Column(JSON, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Thumbnail information
    thumbnail = Column(String, nullable=True)  # Path or base64 encoded thumbnail
    has_thumbnail = Column(
        Boolean, default=False
    )  # Flag to indicate if thumbnail was generated
    thumbnail_generated_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization = relationship("Organization", back_populates="dicom_files")


###
# Pydantic Models for API
###


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user_id: int
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Organization Pydantic Models
class OrganizationCreate(BaseModel):
    name: str
    slug: str

    @validator("slug")
    def validate_slug(cls, v):
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(
                "Slug must be lowercase, contain only alphanumeric characters and hyphens, and cannot start or end with a hyphen"
            )
        return v


class OrganizationResponse(BaseModel):
    id: int
    name: str
    slug: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OrganizationUpdate(BaseModel):
    name: str = None
    description: str = None


class UserOrganizationResponse(BaseModel):
    organization_id: int
    user_id: int
    is_admin: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    email_verified: bool
    picture: str
    organizations: List[OrganizationResponse] = []

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str


class UserUpdate(BaseModel):
    first_name: str = None
    last_name: str = None
    email: str = None
    picture: str = None
    is_admin: bool = None


class ItemCreate(BaseModel):
    title: str
    description: str = None
    data: dict = None


class ItemResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    title: str
    description: str = None
    data: dict = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ItemUpdate(BaseModel):
    title: str = None
    description: str = None
    data: dict = None
    is_active: bool = None


class BatchCreate(BaseModel):
    name: str
    description: str = None
    project_id: int | None = None


class BatchResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    project_id: int | None = None
    name: str
    description: str = None
    processing_status: str | None = None
    created_at: datetime
    updated_at: datetime
    quality_summary: dict = None

    model_config = {"from_attributes": True}


class BatchUpdate(BaseModel):
    name: str = None
    description: str = None
    project_id: int = None
    processing_status: str = None


class ProjectCreate(BaseModel):
    name: str
    description: str = None
    data_region: str = DataRegion.LOCAL.value
    s3_bucket_name: str = None

    @validator("name")
    def validate_project_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9-]+$", v):
            raise ValueError(
                "Project name must contain only alphanumeric characters and hyphens"
            )
        return v


class ProjectResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    name: str
    description: str = None
    data_region: str
    s3_bucket_name: str = None
    first_batch_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectUpdate(BaseModel):
    name: str = None
    description: str = None
    data_region: str = None
    s3_bucket_name: str = None

    @validator("name")
    def validate_project_name(cls, v):
        if v is not None and not re.match(r"^[a-zA-Z0-9-]+$", v):
            raise ValueError(
                "Project name must contain only alphanumeric characters and hyphens"
            )
        return v


class SyntheticDatasetCreate(BaseModel):
    name: str
    description: str = None
    num_patients: int
    batch_id: int = None


class SyntheticDatasetResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    name: str
    description: str = None
    num_patients: int
    batch_id: int = None
    column_mappings: dict = None
    applied_checks: dict = None
    check_results: dict = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SyntheticDatasetUpdate(BaseModel):
    name: str = None
    description: str = None
    batch_id: int = None
    column_mappings: dict = None
    applied_checks: dict = None


class CheckCreate(BaseModel):
    name: str
    description: str = None
    data_type: DataType
    scope: CheckScope = CheckScope.FIELD
    parameters: dict = None
    implementation: str
    python_script: str = None


class CheckResponse(BaseModel):
    id: int
    organization_id: int
    name: str
    description: str = None
    data_type: DataType
    scope: CheckScope
    parameters: dict = None
    implementation: str
    python_script: str = ""  # Default to empty string instead of None
    is_system: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    # Validator to ensure python_script is always a string
    @field_validator("python_script", mode="before")
    @classmethod
    def ensure_string(cls, v):
        if v is None:
            return ""
        return v


class CheckUpdate(BaseModel):
    name: str = None
    description: str = None
    scope: CheckScope = None
    parameters: dict = None
    implementation: str = None
    python_script: str = None
    is_system: bool = None


class ColumnMappingCreate(BaseModel):
    column_name: str
    data_type: DataType
    description: str = None
    synonyms: list[str] = None


class ColumnMappingResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    column_name: str
    data_type: DataType
    description: str = None
    synonyms: list[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ColumnMappingUpdate(BaseModel):
    data_type: DataType = None
    description: str = None
    synonyms: list[str] = None


class ModelCreate(BaseModel):
    name: str
    description: str = None
    version: str
    parameters: dict = None
    implementation: str


class ModelResponse(BaseModel):
    id: int
    organization_id: int
    name: str
    description: str = None
    version: str
    parameters: dict = None
    implementation: str
    is_system: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelUpdate(BaseModel):
    name: str = None
    description: str = None
    version: str = None
    parameters: dict = None
    implementation: str = None
    is_system: bool = None


class ModelImplementationsResponse(BaseModel):
    implementations: Dict[str, str]


class ModelPrediction(BaseModel):
    label: str
    probability: float


class ModelEvaluationResults(BaseModel):
    confidence: float
    predictions: List[ModelPrediction]


class ModelEvaluationResponse(BaseModel):
    compatible: bool
    file_id: int
    model_id: int
    reason: Optional[str] = None
    evaluation_results: Optional[Dict[str, Any]] = None


class SyntheticPatient(BaseModel):
    patient_ngsci_id: str
    year: int
    verbal_consent: str
    age: int
    sex: str
    bp: str
    bp_systolic: float
    bp_diastolic: float
    pulse: str
    pulse_entry: float
    resp_rate: str
    resp_rate_entry: float
    spo2: str
    spo2_entry: float
    rbs: str
    rbs_entry: float
    height: str
    height_entry: float
    weight: str
    weight_entry: float
    midarm_circum: str
    midarm_circum_entry: float
    waist_circum: str
    waist_circum_entry: float
    hip_circum: str
    hip_circum_entry: float
    endurance_test: str
    endurance_test_entry: float
    grip_left: str
    grip_left_entry: float
    grip_right: str
    grip_right_entry: float
    tonometry_lefteye: str
    tonometry_lefteye_entry: float
    tonometry_righteye: str
    tonometry_righteye_entry: float
    fundus_lefteye: str
    fundus_lefteye_obs: str = None
    fundus_righteye: str
    fundus_righteye_obs: str = None
    cognition_sf: str
    cognition_sf_score: float
    cognit_impaired: int
    Hb: float
    HbA1c: float
    triglycerides_mg_dl: float
    tot_cholesterol_mg_dl: float
    HDL_mg_dl: float
    LDL_mg_dl: float
    VLDL_mg_dl: float
    totchol_by_hdl_ratio: float
    ldl_by_hdl_ratio: float
    creatinine_mg_dl: float
    literate: str
    smoking_1: str
    smoking_2: str = None
    smoking_3: str
    phq_1: str
    phq_2: str
    phq_3: str
    phq_4: str
    direct_lonely: str


# DICOM Pydantic Models
class DicomFileCreate(BaseModel):
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    patient_id: str | None = None
    patient_name: str | None = None
    batch_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    modality: str | None = None
    batch_date: datetime | None = None
    dicom_metadata: dict | None = None

    model_config = {"arbitrary_types_allowed": True}


class DicomFileResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    patient_id: str | None = None
    patient_name: str | None = None
    batch_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    modality: str | None = None
    batch_date: datetime | None = None
    dicom_metadata: dict | None = None
    processing_status: str
    processing_results: dict | None = None
    processed_at: datetime | None = None
    thumbnail: str | None = None
    has_thumbnail: bool
    thumbnail_generated_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}


# File Pydantic Models
class FileCreate(BaseModel):
    batch_id: int
    project_id: int | None = None
    file_type: str | None = None  # If None, will be auto-detected from file extension
    file_metadata: dict | None = None

    model_config = {"arbitrary_types_allowed": True}


class FileResponse(BaseModel):
    id: int
    organization_id: int
    user_id: int
    batch_id: int
    project_id: int | None = None
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    file_type: str
    file_metadata: dict | None = None
    csv_headers: list | None = None
    potential_mappings: dict | None = None
    schema_type: str | None = None
    processing_status: str
    processing_results: dict | None = None
    processed_at: datetime | None = None
    thumbnail: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}


class FileUpdate(BaseModel):
    file_type: str | None = None
    file_metadata: dict | None = None
    potential_mappings: dict | None = None
    schema_type: str | None = None
    thumbnail: str | None = None
    has_thumbnail: bool | None = None
    project_id: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class ECGAnalysisResponse(BaseModel):
    id: int
    file_id: int
    has_missing_leads: bool | None = None
    signal_noise_ratio: float | None = None
    baseline_wander_score: float | None = None
    motion_artifact_score: float | None = None
    rr_interval_mean: float | None = None
    rr_interval_stddev: float | None = None
    rr_interval_consistency: float | None = None
    qrs_count: int | None = None
    qrs_detection_confidence: float | None = None
    hrv_sdnn: float | None = None
    hrv_rmssd: float | None = None
    hrv_pnn50: float | None = None
    hrv_lf: float | None = None
    hrv_hf: float | None = None
    hrv_lf_hf_ratio: float | None = None
    frequency_peak: float | None = None
    frequency_power_vlf: float | None = None
    frequency_power_lf: float | None = None
    frequency_power_hf: float | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
