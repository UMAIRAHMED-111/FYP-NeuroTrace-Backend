from sqlalchemy import Column, String, Enum as SQLEnum, DateTime, UUID
from sqlalchemy.sql import func
from app.database import Base
import uuid
from enum import Enum as PyEnum

class UserRoleDB(PyEnum):
    patient = 'patient'
    caregiver = 'caregiver'

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRoleDB), nullable=False, default=UserRoleDB.patient)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<User {self.email}>"