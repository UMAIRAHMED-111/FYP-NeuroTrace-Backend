from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID 

class CaregiverProfileBase(BaseModel):
    full_name: Optional[str] = None
    contact_info: Optional[str] = None
    relationship: Optional[str] = None
    alert_prefs: Optional[Dict[str, Any]] = None
    permissions: Optional[str] = None

class CaregiverProfileCreate(CaregiverProfileBase):
    user_id: str

class CaregiverProfile(CaregiverProfileBase):
    user_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
class CaregiverProfileUpdate(CaregiverProfileBase):
    pass  # Inherits all optional fields
