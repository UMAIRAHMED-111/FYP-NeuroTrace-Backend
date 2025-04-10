from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User, UserRoleDB
from app.models.caregiver import CaregiverProfile
from app.schemas.link import LinkedCaregiver  # reusing existing combined schema

router = APIRouter(tags=["Caregivers"])

@router.get("/", response_model=list[LinkedCaregiver])
def get_all_caregivers(db: Session = Depends(get_db)):
    caregivers = []

    users = db.query(User).filter(User.role == UserRoleDB.caregiver).all()
    for user in users:
        profile = db.query(CaregiverProfile).filter_by(user_id=user.id).first()
        if profile:
            caregivers.append({
                "user": user,
                "profile": profile
            })

    return caregivers
