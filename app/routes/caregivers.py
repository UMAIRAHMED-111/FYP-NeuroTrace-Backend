from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.caregiver import CaregiverProfile
from app.schemas.caregiver import CaregiverProfile, CaregiverProfileCreate
from app.utils.security import get_current_user

router = APIRouter(tags=["Caregivers"])
