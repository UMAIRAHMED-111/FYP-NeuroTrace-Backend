from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional

class ReminderCreate(BaseModel):
    user_id: UUID
    title: str
    description: Optional[str] = None
    scheduled_for: datetime

class ReminderOut(ReminderCreate):
    id: UUID
    is_completed: bool
    created_at: datetime

    class Config:
        from_attributes = True
