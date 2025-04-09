from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional

class NotificationCreate(BaseModel):
    user_id: UUID
    title: str
    message: str
    type: Optional[str] = "general"

class NotificationOut(NotificationCreate):
    id: UUID
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True
