from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from app.database import get_db
from app.models.user import User as UserModel
from app.models.user import User, UserRoleDB
from app.utils.security import get_current_user
from app.schemas.user import UserCreate, UserOut, TokenRequest, RegisterRequest, UserWithProfile, UserWithProfileUpdate
from app.utils.security import (
    get_password_hash,
    create_access_token,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
import uuid
from app.services.user_service import register_user_with_profile, update_user_and_profile
from app.models.patient import PatientProfile
from app.models.caregiver import CaregiverProfile

router = APIRouter(tags=["Authentication"])

@router.post("/register", response_model=UserOut)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user = register_user_with_profile(db, request)
    return user

@router.post("/token")
async def login_for_access_token(
    request: TokenRequest,  # Use the new model instead of OAuth2PasswordRequestForm
    db: Session = Depends(get_db)
):
    try:
        user = db.query(UserModel).filter(UserModel.email == request.username).first()
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user.id),
            "role": user.role.value
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
        
@router.get("/profile", response_model=UserWithProfile)
def get_me(
    db: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    profile = None
    if user.role == UserRoleDB.patient:
        profile = db.query(PatientProfile).filter_by(user_id=user.id).first()
    elif user.role == UserRoleDB.caregiver:
        profile = db.query(CaregiverProfile).filter_by(user_id=user.id).first()

    return {"user": user, "profile": profile}

@router.patch("/update", response_model=UserWithProfile)
def patch_me(
    payload: UserWithProfileUpdate,
    db: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user)
):
    updated = update_user_and_profile(db, user_id, payload.dict(exclude_unset=True))
    return updated