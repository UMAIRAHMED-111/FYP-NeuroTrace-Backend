from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.database import get_db
from app.models.notification import Notification
from app.schemas.notification import NotificationCreate, NotificationOut
from app.utils.security import get_current_user

router = APIRouter(tags=["Notifications"])


@router.post("/", response_model=NotificationOut)
def create_notification(
    payload: NotificationCreate,
    db: Session = Depends(get_db),
):
    notification = Notification(**payload.dict())
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


@router.get("/", response_model=list[NotificationOut])
def get_notifications(
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    return db.query(Notification).filter_by(user_id=current_user_id).order_by(Notification.created_at.desc()).all()


@router.get("/{id}", response_model=NotificationOut)
def get_notification_by_id(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    notif = db.query(Notification).filter_by(id=id, user_id=current_user_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notif


@router.patch("/{id}", response_model=NotificationOut)
def mark_as_read(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    notif = db.query(Notification).filter_by(id=id, user_id=current_user_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")

    notif.is_read = True
    db.commit()
    db.refresh(notif)
    return notif


@router.delete("/{id}", response_model=dict)
def delete_notification(
    id: UUID,
    db: Session = Depends(get_db),
    current_user_id: UUID = Depends(get_current_user),
):
    notif = db.query(Notification).filter_by(id=id, user_id=current_user_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")

    db.delete(notif)
    db.commit()
    return {"message": "Notification deleted"}
