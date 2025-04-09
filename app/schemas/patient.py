from pydantic import BaseModel, ConfigDict
from datetime import date, datetime
from typing import Optional, Dict, Any
from uuid import UUID 

class PatientProfileBase(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    profile_info: Optional[str] = None
    geofence_config: Optional[Dict[str, Any]] = None

class PatientProfileCreate(PatientProfileBase):
    user_id: str

class PatientProfile(PatientProfileBase):
    user_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
class PatientProfileUpdate(PatientProfileBase):
    pass  # Inherits all optional fields