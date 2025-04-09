from sqlalchemy import Column, String, Date, Text, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base

class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    user_id = Column(UUID(as_uuid=True), primary_key=True)
    full_name = Column(String(100))
    phone = Column(String(20))
    date_of_birth = Column(Date)
    profile_info = Column(Text)
    geofence_config = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<PatientProfile {self.full_name}>"