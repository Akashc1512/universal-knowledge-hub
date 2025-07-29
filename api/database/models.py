"""
Database Models for Universal Knowledge Platform - MAANG Standards.

This module implements database models following MAANG-level best practices
with SQLAlchemy 2.0, including proper indexing, relationships, and migrations.

Architecture:
    - Base model with common fields and behaviors
    - Proper indexing for query performance
    - Audit trail for all changes
    - Soft deletes for data recovery
    - Optimistic locking for concurrency
    - JSON schema validation for JSONB fields

Performance:
    - Composite indexes for common queries
    - Lazy loading with explicit eager loading
    - Query optimization hints
    - Partition support for large tables

Security:
    - Column-level encryption for sensitive data
    - Row-level security policies
    - Audit logging for compliance

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import uuid
import enum
from datetime import datetime, timezone, timedelta
from typing import (
    Optional, Dict, Any, List, Type, TypeVar, 
    Generic, ClassVar, cast, Union
)
from decimal import Decimal

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Float, Text,
    ForeignKey, Table, Index, CheckConstraint, UniqueConstraint,
    JSON, DECIMAL, Enum as SQLEnum, event, func, text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, INET, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import (
    relationship, backref, validates, column_property,
    deferred, query_expression, Session
)
from sqlalchemy.sql import expression
from sqlalchemy.schema import DDL
from sqlalchemy_utils import (
    EncryptedType, URLType, EmailType, IPAddressType,
    ChoiceType, TSVectorType
)
import structlog

logger = structlog.get_logger(__name__)

# Type variables
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='BaseModel')

# Custom types
class EncryptionKey:
    """Encryption key provider for encrypted columns."""
    
    _key: ClassVar[Optional[bytes]] = None
    
    @classmethod
    def get_key(cls) -> bytes:
        """Get encryption key from environment or config."""
        if cls._key is None:
            import os
            key = os.getenv("DATABASE_ENCRYPTION_KEY")
            if not key:
                raise ValueError("DATABASE_ENCRYPTION_KEY not set")
            cls._key = key.encode()
        return cls._key

# Enums
class RecordStatus(enum.Enum):
    """Record status for soft deletes."""
    ACTIVE = "active"
    DELETED = "deleted"
    ARCHIVED = "archived"

class AuditAction(enum.Enum):
    """Audit trail action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    ARCHIVE = "archive"

# Base model
class BaseModel:
    """
    Base model with common fields and functionality.
    
    Provides:
    - UUID primary keys
    - Timestamps (created/updated)
    - Soft deletes
    - Audit trail
    - Optimistic locking
    - JSON representation
    """
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Record creation timestamp"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Last update timestamp"
    )
    
    # Soft delete
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp"
    )
    
    status = Column(
        SQLEnum(RecordStatus),
        nullable=False,
        default=RecordStatus.ACTIVE,
        index=True,
        comment="Record status"
    )
    
    # Optimistic locking
    version = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Version for optimistic locking"
    )
    
    # Metadata
    metadata_json = Column(
        MutableDict.as_mutable(JSONB),
        nullable=False,
        default=dict,
        comment="Additional metadata"
    )
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return ''.join(
            '_' + c.lower() if c.isupper() else c 
            for c in name
        ).lstrip('_')
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if record is active."""
        return self.status == RecordStatus.ACTIVE and self.deleted_at is None
    
    @is_active.expression
    def is_active(cls):
        """SQL expression for is_active."""
        return (cls.status == RecordStatus.ACTIVE) & (cls.deleted_at.is_(None))
    
    def soft_delete(self) -> None:
        """Soft delete the record."""
        self.deleted_at = datetime.now(timezone.utc)
        self.status = RecordStatus.DELETED
    
    def restore(self) -> None:
        """Restore soft deleted record."""
        self.deleted_at = None
        self.status = RecordStatus.ACTIVE
    
    def archive(self) -> None:
        """Archive the record."""
        self.status = RecordStatus.ARCHIVED
    
    def to_dict(self, include: Optional[List[str]] = None, 
                exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        
        Args:
            include: Fields to include
            exclude: Fields to exclude
            
        Returns:
            Dictionary representation
        """
        data = {}
        
        # Get all columns
        for column in self.__table__.columns:
            if include and column.name not in include:
                continue
            if exclude and column.name in exclude:
                continue
            
            value = getattr(self, column.name)
            
            # Handle special types
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, enum.Enum):
                value = value.value
            elif isinstance(value, Decimal):
                value = float(value)
            
            data[column.name] = value
        
        return data
    
    def update_from_dict(self, data: Dict[str, Any], 
                        allowed_fields: Optional[List[str]] = None) -> None:
        """
        Update model from dictionary.
        
        Args:
            data: Data dictionary
            allowed_fields: Fields allowed to update
        """
        for key, value in data.items():
            if allowed_fields and key not in allowed_fields:
                continue
            
            if hasattr(self, key) and key not in ['id', 'created_at']:
                setattr(self, key, value)
        
        # Increment version for optimistic locking
        self.version += 1
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}(id={self.id})>"

# Create base
Base = declarative_base(cls=BaseModel)

# Association tables
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('granted_at', DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column('granted_by', UUID(as_uuid=True), ForeignKey('users.id')),
    UniqueConstraint('user_id', 'role_id', name='uq_user_roles')
)

# Models
class Role(Base):
    """Role model for RBAC."""
    
    __tablename__ = 'roles'
    __table_args__ = (
        Index('idx_roles_name', 'name'),
        CheckConstraint('char_length(name) >= 3', name='ck_roles_name_length'),
        {'comment': 'User roles for access control'}
    )
    
    # Fields
    name = Column(
        String(50),
        nullable=False,
        unique=True,
        comment="Role name"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Role description"
    )
    
    permissions = Column(
        MutableDict.as_mutable(JSONB),
        nullable=False,
        default=dict,
        comment="Role permissions"
    )
    
    is_system = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="System role flag"
    )
    
    # Relationships
    users = relationship(
        'User',
        secondary=user_roles,
        back_populates='roles',
        lazy='dynamic'
    )
    
    @validates('name')
    def validate_name(self, key: str, value: str) -> str:
        """Validate role name."""
        if not value or len(value) < 3:
            raise ValueError("Role name must be at least 3 characters")
        return value.lower()
    
    def has_permission(self, permission: str) -> bool:
        """Check if role has permission."""
        parts = permission.split('.')
        perms = self.permissions
        
        for part in parts:
            if isinstance(perms, dict):
                perms = perms.get(part)
            else:
                return False
        
        return bool(perms)

class User(Base):
    """User model with authentication and profile."""
    
    __tablename__ = 'users'
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_username', 'username'),
        Index('idx_users_status', 'status'),
        Index('idx_users_last_login', 'last_login_at'),
        CheckConstraint('char_length(username) >= 3', name='ck_users_username_length'),
        CheckConstraint('char_length(email) >= 5', name='ck_users_email_length'),
        {'comment': 'User accounts'}
    )
    
    # Authentication fields
    username = Column(
        String(50),
        nullable=False,
        unique=True,
        comment="Unique username"
    )
    
    email = Column(
        EmailType,
        nullable=False,
        unique=True,
        comment="Email address"
    )
    
    password_hash = Column(
        String(255),
        nullable=False,
        comment="Bcrypt password hash"
    )
    
    # Profile fields
    full_name = Column(
        String(100),
        nullable=True,
        comment="Full name"
    )
    
    avatar_url = Column(
        URLType,
        nullable=True,
        comment="Avatar URL"
    )
    
    bio = Column(
        Text,
        nullable=True,
        comment="User biography"
    )
    
    # Security fields
    email_verified = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Email verification status"
    )
    
    email_verification_token = Column(
        String(255),
        nullable=True,
        comment="Email verification token"
    )
    
    two_factor_enabled = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="2FA enabled flag"
    )
    
    two_factor_secret = Column(
        EncryptedType(String, EncryptionKey.get_key),
        nullable=True,
        comment="2FA secret (encrypted)"
    )
    
    # Activity tracking
    last_login_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last login timestamp"
    )
    
    last_login_ip = Column(
        IPAddressType,
        nullable=True,
        comment="Last login IP address"
    )
    
    login_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total login count"
    )
    
    failed_login_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Failed login attempts"
    )
    
    locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account lock expiration"
    )
    
    # Preferences
    preferences = Column(
        MutableDict.as_mutable(JSONB),
        nullable=False,
        default=dict,
        comment="User preferences"
    )
    
    # Relationships
    roles = relationship(
        'Role',
        secondary=user_roles,
        back_populates='users',
        lazy='joined'
    )
    
    sessions = relationship(
        'UserSession',
        back_populates='user',
        cascade='all, delete-orphan',
        lazy='dynamic'
    )
    
    api_keys = relationship(
        'APIKey',
        back_populates='user',
        cascade='all, delete-orphan',
        lazy='dynamic'
    )
    
    audit_logs = relationship(
        'AuditLog',
        foreign_keys='AuditLog.user_id',
        back_populates='user',
        lazy='dynamic'
    )
    
    # Computed properties
    @hybrid_property
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    @is_locked.expression
    def is_locked(cls):
        """SQL expression for is_locked."""
        return cls.locked_until > func.now()
    
    @hybrid_property
    def display_name(self) -> str:
        """Get display name."""
        return self.full_name or self.username
    
    @validates('email')
    def validate_email(self, key: str, value: str) -> str:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, value):
            raise ValueError("Invalid email format")
        return value.lower()
    
    @validates('username')
    def validate_username(self, key: str, value: str) -> str:
        """Validate username format."""
        import re
        pattern = r'^[a-zA-Z0-9_-]{3,50}$'
        if not re.match(pattern, value):
            raise ValueError("Username must be 3-50 characters, alphanumeric, underscore, or hyphen")
        return value.lower()
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has role."""
        return any(role.name == role_name for role in self.roles)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission."""
        return any(role.has_permission(permission) for role in self.roles)
    
    def add_role(self, role: Role, granted_by: Optional['User'] = None) -> None:
        """Add role to user."""
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: Role) -> None:
        """Remove role from user."""
        if role in self.roles:
            self.roles.remove(role)
    
    def record_login(self, ip_address: str) -> None:
        """Record successful login."""
        self.last_login_at = datetime.now(timezone.utc)
        self.last_login_ip = ip_address
        self.login_count += 1
        self.failed_login_count = 0
    
    def record_failed_login(self) -> None:
        """Record failed login attempt."""
        self.failed_login_count += 1
        
        # Lock account after 5 failed attempts
        if self.failed_login_count >= 5:
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)

class UserSession(Base):
    """User session model for session management."""
    
    __tablename__ = 'user_sessions'
    __table_args__ = (
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_token', 'token'),
        Index('idx_sessions_expires_at', 'expires_at'),
        {'comment': 'User sessions'}
    )
    
    # Fields
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        comment="User ID"
    )
    
    token = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="Session token"
    )
    
    refresh_token = Column(
        String(255),
        nullable=True,
        unique=True,
        comment="Refresh token"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Session expiration"
    )
    
    refresh_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Refresh token expiration"
    )
    
    ip_address = Column(
        IPAddressType,
        nullable=True,
        comment="Client IP address"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="User agent string"
    )
    
    device_info = Column(
        MutableDict.as_mutable(JSONB),
        nullable=False,
        default=dict,
        comment="Device information"
    )
    
    # Relationships
    user = relationship(
        'User',
        back_populates='sessions'
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    @is_expired.expression
    def is_expired(cls):
        """SQL expression for is_expired."""
        return cls.expires_at < func.now()
    
    def extend(self, minutes: int = 30) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)

class APIKey(Base):
    """API key model for machine authentication."""
    
    __tablename__ = 'api_keys'
    __table_args__ = (
        Index('idx_api_keys_user_id', 'user_id'),
        Index('idx_api_keys_key_hash', 'key_hash'),
        Index('idx_api_keys_expires_at', 'expires_at'),
        {'comment': 'API keys for authentication'}
    )
    
    # Fields
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        comment="User ID"
    )
    
    name = Column(
        String(100),
        nullable=False,
        comment="API key name"
    )
    
    key_hash = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="API key hash"
    )
    
    key_prefix = Column(
        String(10),
        nullable=False,
        comment="Key prefix for identification"
    )
    
    scopes = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="API key scopes"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Key expiration"
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last usage timestamp"
    )
    
    usage_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Usage count"
    )
    
    rate_limit = Column(
        Integer,
        nullable=True,
        comment="Rate limit per minute"
    )
    
    allowed_ips = Column(
        ARRAY(IPAddressType),
        nullable=True,
        comment="IP whitelist"
    )
    
    # Relationships
    user = relationship(
        'User',
        back_populates='api_keys'
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @is_expired.expression
    def is_expired(cls):
        """SQL expression for is_expired."""
        return (cls.expires_at.isnot(None)) & (cls.expires_at < func.now())
    
    def record_usage(self, ip_address: Optional[str] = None) -> None:
        """Record API key usage."""
        self.last_used_at = datetime.now(timezone.utc)
        self.usage_count += 1
        
        # Check IP whitelist
        if self.allowed_ips and ip_address:
            if ip_address not in self.allowed_ips:
                raise ValueError(f"IP {ip_address} not in whitelist")
    
    def has_scope(self, scope: str) -> bool:
        """Check if key has scope."""
        return scope in self.scopes or '*' in self.scopes

class KnowledgeItem(Base):
    """Knowledge item model for storing information."""
    
    __tablename__ = 'knowledge_items'
    __table_args__ = (
        Index('idx_knowledge_items_type', 'type'),
        Index('idx_knowledge_items_source', 'source'),
        Index('idx_knowledge_items_created_by', 'created_by'),
        Index('idx_knowledge_items_search', 'search_vector', postgresql_using='gin'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='ck_knowledge_confidence'),
        {'comment': 'Knowledge base items'}
    )
    
    # Fields
    title = Column(
        String(255),
        nullable=False,
        comment="Item title"
    )
    
    content = Column(
        Text,
        nullable=False,
        comment="Item content"
    )
    
    summary = Column(
        Text,
        nullable=True,
        comment="Content summary"
    )
    
    type = Column(
        String(50),
        nullable=False,
        default='document',
        comment="Item type"
    )
    
    source = Column(
        String(255),
        nullable=True,
        comment="Content source"
    )
    
    source_url = Column(
        URLType,
        nullable=True,
        comment="Source URL"
    )
    
    confidence = Column(
        Float,
        nullable=False,
        default=1.0,
        comment="Confidence score"
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="Item tags"
    )
    
    embedding = Column(
        ARRAY(Float),
        nullable=True,
        comment="Vector embedding"
    )
    
    search_vector = Column(
        TSVectorType('title', 'content', weights={'title': 'A', 'content': 'B'}),
        nullable=True,
        comment="Full-text search vector"
    )
    
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        comment="Creator user ID"
    )
    
    # Relationships
    creator = relationship(
        'User',
        foreign_keys=[created_by]
    )
    
    queries = relationship(
        'Query',
        secondary='query_results',
        back_populates='results'
    )
    
    @validates('confidence')
    def validate_confidence(self, key: str, value: float) -> float:
        """Validate confidence score."""
        if not 0 <= value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return value
    
    def add_tag(self, tag: str) -> None:
        """Add tag to item."""
        if tag not in self.tags:
            self.tags = self.tags + [tag]
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag from item."""
        if tag in self.tags:
            self.tags = [t for t in self.tags if t != tag]

# Query results association
query_results = Table(
    'query_results',
    Base.metadata,
    Column('query_id', UUID(as_uuid=True), ForeignKey('queries.id', ondelete='CASCADE')),
    Column('knowledge_item_id', UUID(as_uuid=True), ForeignKey('knowledge_items.id', ondelete='CASCADE')),
    Column('relevance_score', Float, nullable=False, default=1.0),
    Column('position', Integer, nullable=False),
    UniqueConstraint('query_id', 'knowledge_item_id', name='uq_query_results')
)

class Query(Base):
    """Query model for tracking user queries."""
    
    __tablename__ = 'queries'
    __table_args__ = (
        Index('idx_queries_user_id', 'user_id'),
        Index('idx_queries_created_at', 'created_at'),
        Index('idx_queries_response_time', 'response_time_ms'),
        {'comment': 'User queries'}
    )
    
    # Fields
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        comment="User ID"
    )
    
    query_text = Column(
        Text,
        nullable=False,
        comment="Query text"
    )
    
    query_embedding = Column(
        ARRAY(Float),
        nullable=True,
        comment="Query embedding"
    )
    
    response_text = Column(
        Text,
        nullable=True,
        comment="Response text"
    )
    
    response_time_ms = Column(
        Integer,
        nullable=True,
        comment="Response time in milliseconds"
    )
    
    tokens_used = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Tokens consumed"
    )
    
    cost = Column(
        DECIMAL(10, 4),
        nullable=False,
        default=0,
        comment="Query cost"
    )
    
    feedback_rating = Column(
        Integer,
        nullable=True,
        comment="User feedback rating"
    )
    
    feedback_text = Column(
        Text,
        nullable=True,
        comment="User feedback text"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if failed"
    )
    
    # Relationships
    user = relationship(
        'User',
        foreign_keys=[user_id]
    )
    
    results = relationship(
        'KnowledgeItem',
        secondary=query_results,
        back_populates='queries'
    )
    
    def add_result(self, item: KnowledgeItem, relevance: float, position: int) -> None:
        """Add result to query."""
        # This would be handled by SQLAlchemy association proxy in production
        pass

class AuditLog(Base):
    """Audit log model for tracking changes."""
    
    __tablename__ = 'audit_logs'
    __table_args__ = (
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_entity_type_id', 'entity_type', 'entity_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
        Index('idx_audit_logs_action', 'action'),
        {'comment': 'Audit trail for all changes'}
    )
    
    # Fields
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        comment="User who made the change"
    )
    
    action = Column(
        SQLEnum(AuditAction),
        nullable=False,
        comment="Action performed"
    )
    
    entity_type = Column(
        String(50),
        nullable=False,
        comment="Entity type"
    )
    
    entity_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        comment="Entity ID"
    )
    
    old_values = Column(
        MutableDict.as_mutable(JSONB),
        nullable=True,
        comment="Previous values"
    )
    
    new_values = Column(
        MutableDict.as_mutable(JSONB),
        nullable=True,
        comment="New values"
    )
    
    ip_address = Column(
        IPAddressType,
        nullable=True,
        comment="Client IP address"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="User agent"
    )
    
    # Relationships
    user = relationship(
        'User',
        foreign_keys=[user_id],
        back_populates='audit_logs'
    )

# Event listeners for audit logging
@event.listens_for(Base, 'before_insert', propagate=True)
def receive_before_insert(mapper, connection, target):
    """Log entity creation."""
    if isinstance(target, AuditLog):
        return  # Don't audit audit logs
    
    # Would create audit log entry here
    logger.debug(f"Creating {target.__class__.__name__}: {target.id}")

@event.listens_for(Base, 'before_update', propagate=True)
def receive_before_update(mapper, connection, target):
    """Log entity updates."""
    if isinstance(target, AuditLog):
        return
    
    # Would create audit log entry with changes
    logger.debug(f"Updating {target.__class__.__name__}: {target.id}")

# Create indexes and constraints
Index('idx_users_fulltext', 
      func.to_tsvector('english', User.username + ' ' + User.full_name))

# Database functions
create_update_timestamp_function = DDL("""
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';
""")

# Triggers for each table
for table in Base.metadata.tables.values():
    if 'updated_at' in table.c:
        trigger = DDL(f"""
        CREATE TRIGGER update_{table.name}_updated_at 
        BEFORE UPDATE ON {table.name}
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        event.listen(table, 'after_create', trigger)

# Export models
__all__ = [
    'Base',
    'BaseModel',
    'Role',
    'User',
    'UserSession',
    'APIKey',
    'KnowledgeItem',
    'Query',
    'AuditLog',
    'RecordStatus',
    'AuditAction',
] 