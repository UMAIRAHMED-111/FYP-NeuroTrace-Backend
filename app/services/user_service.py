# services/user_service.py

from sqlalchemy.orm import Session
from app.models.user import User, UserRoleDB
from app.models.patient import PatientProfile
from app.models.caregiver import CaregiverProfile
from app.schemas.user import RegisterRequest
from app.utils.security import get_password_hash
from fastapi import HTTPException, status
import uuid


def register_user_with_profile(db: Session, data: RegisterRequest) -> User:
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = get_password_hash(data.password)
    user = User(email=data.email, password_hash=hashed_pw, role=data.role)
    db.add(user)
    db.commit()
    db.refresh(user)

    if user.role == UserRoleDB.patient:
        profile = PatientProfile(user_id=user.id, **data.profile.dict())
        db.add(profile)

    elif user.role == UserRoleDB.caregiver:
        profile = CaregiverProfile(user_id=user.id, **data.profile.dict())
        db.add(profile)

    db.commit()
    return user

def update_user_and_profile(
    db: Session,
    user_id: uuid.UUID,
    update_data: dict
):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = update_data.get("user", {})
    profile_data = update_data.get("profile", {})

    # Update user fields
    for key, value in user_data.items():
        if value is not None:
            if key == "password":
                user.password_hash = get_password_hash(value)  # ✅ Hash password securely
            else:
                setattr(user, key, value)

    # Update profile
    if user.role == UserRoleDB.patient:
        profile = db.query(PatientProfile).filter_by(user_id=user.id).first()
    elif user.role == UserRoleDB.caregiver:
        profile = db.query(CaregiverProfile).filter_by(user_id=user.id).first()
    else:
        raise HTTPException(status_code=400, detail="Unknown user role")

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    for key, value in profile_data.items():
        if value is not None:
            setattr(profile, key, value)

    db.commit()
    db.refresh(user)
    db.refresh(profile)

    return {
        "user": user,
        "profile": profile
    }