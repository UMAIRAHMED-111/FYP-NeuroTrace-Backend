from pydantic import BaseModel, EmailStr, ConfigDict, field_validator
from .patient import PatientProfileBase
from .caregiver import CaregiverProfileBase
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
from app.schemas.patient import PatientProfile as PatientOut
from app.schemas.caregiver import CaregiverProfile as CaregiverOut
from app.schemas.patient import PatientProfileUpdate
from app.schemas.caregiver import CaregiverProfileUpdate
from typing import Union

class UserRole(str, Enum):
    PATIENT = 'patient'
    CAREGIVER = 'caregiver'

class UserBase(BaseModel):
    email: EmailStr
    role: UserRole

class UserCreate(UserBase):
    password: str
    
    @field_validator('role', mode='before')
    def validate_role(cls, v):
        if isinstance(v, str):
            v = v.lower()
            for role in UserRole:
                if v == role.value:
                    return role
        raise ValueError(f"Invalid role. Must be one of: {[role.value for role in UserRole]}")

class UserOut(UserBase):
    id: uuid.UUID  # Changed to UUID type
    created_at: datetime
    last_login_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class RegisterRequest(UserCreate):
    profile: PatientProfileBase | CaregiverProfileBase
    
class TokenRequest(BaseModel):
    username: str
    password: str
    
class UserWithProfile(BaseModel):
    user: UserOut
    profile: Union[PatientOut, CaregiverOut, None]
    
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    
class UserWithProfileUpdate(BaseModel):
    user: UserUpdate
    profile: PatientProfileUpdate | CaregiverProfileUpdate