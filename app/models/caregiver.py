from sqlalchemy import Column, String, Text, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base

class CaregiverProfile(Base):
    __tablename__ = "caregiver_profiles"

    user_id = Column(UUID(as_uuid=True), primary_key=True)
    full_name = Column(String(100))
    contact_info = Column(Text)
    relationship = Column(String(50))
    alert_prefs = Column(JSON)
    permissions = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<CaregiverProfile {self.full_name}>"