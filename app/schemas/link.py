from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime
from app.schemas.user import UserOut
from app.schemas.patient import PatientProfile
from app.schemas.caregiver import CaregiverProfile

class LinkedCaregiver(BaseModel):
    user: UserOut
    profile: CaregiverProfile

class LinkedPatient(BaseModel):
    user: UserOut
    profile: PatientProfile

class CaregiverPatientLink(BaseModel):
    id: UUID
    caregiver_id: UUID
    patient_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LinkRequest(BaseModel):
    other_user_id: UUID
