from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.database import get_db
from app.models.reminder import Reminder
from app.schemas.reminder import ReminderCreate, ReminderOut
from app.utils.security import get_current_user

router = APIRouter(tags=["Reminders"])


@router.post("/", response_model=ReminderOut)
def create_reminder(
    payload: ReminderCreate,
    db: Session = Depends(get_db)
):
    reminder = Reminder(**payload.dict())
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.get("/", response_model=list[ReminderOut])
def get_reminders(
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    return db.query(Reminder).filter_by(user_id=current_user_id).order_by(Reminder.scheduled_for).all()


@router.get("/{id}", response_model=ReminderOut)
def get_reminder_by_id(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    reminder = db.query(Reminder).filter_by(id=id, user_id=current_user_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return reminder


@router.patch("/{id}", response_model=ReminderOut)
def mark_reminder_completed(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    reminder = db.query(Reminder).filter_by(id=id, user_id=current_user_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    reminder.is_completed = True
    db.commit()
    db.refresh(reminder)
    return reminder


@router.delete("/{id}", response_model=dict)
def delete_reminder(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    reminder = db.query(Reminder).filter_by(id=id, user_id=current_user_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    db.delete(reminder)
    db.commit()
    return {"message": "Reminder deleted successfully"}
