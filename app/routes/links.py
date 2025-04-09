import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.link import CaregiverPatientLink as CaregiverPatientLinkModel
from app.models.user import User, UserRoleDB
from app.utils.security import get_current_user
from app.schemas.link import CaregiverPatientLink, LinkRequest 
from app.models.caregiver import CaregiverProfile
from app.models.patient import PatientProfile
from app.schemas.link import LinkedPatient
from app.schemas.link import LinkedCaregiver

router = APIRouter(tags=["Relationships"])

@router.post("/", response_model=CaregiverPatientLink)
def create_link(
    link_data: LinkRequest,
    db: Session = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user)
):
    current_user = db.query(User).filter(User.id == current_user_id).first()
    other_user = db.query(User).filter(User.id == link_data.other_user_id).first()

    if not current_user or not other_user:
        raise HTTPException(status_code=404, detail="One or both users not found")

    if current_user.role == other_user.role:
        raise HTTPException(
            status_code=400,
            detail="You can only link a caregiver to a patient"
        )

    # Determine which user is the caregiver/patient
    caregiver_id = current_user_id if current_user.role == UserRoleDB.caregiver else other_user.id
    patient_id = current_user_id if current_user.role == UserRoleDB.patient else other_user.id

    # Prevent duplicates
    existing = db.query(CaregiverPatientLinkModel).filter_by(
        caregiver_id=caregiver_id,
        patient_id=patient_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Link already exists")

    new_link = CaregiverPatientLinkModel(
        caregiver_id=caregiver_id,
        patient_id=patient_id
    )
    db.add(new_link)
    db.commit()
    db.refresh(new_link)
    return new_link

@router.get("/caregivers", response_model=list[LinkedCaregiver])
def list_caregivers_for_patient(
    db: Session = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user)
):
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user or user.role != UserRoleDB.patient:
        raise HTTPException(status_code=403, detail="Only patients can view caregivers")

    links = db.query(CaregiverPatientLinkModel).filter_by(patient_id=user.id).all()
    
    caregivers = []
    for link in links:
        caregiver_user = db.query(User).filter_by(id=link.caregiver_id).first()
        caregiver_profile = db.query(CaregiverProfile).filter_by(user_id=link.caregiver_id).first()
        if caregiver_user and caregiver_profile:
            caregivers.append({
                "user": caregiver_user,
                "profile": caregiver_profile
            })

    return caregivers

@router.get("/patients", response_model=list[LinkedPatient])
def list_patients_for_caregiver(
    db: Session = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user)
):
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user or user.role != UserRoleDB.caregiver:
        raise HTTPException(status_code=403, detail="Only caregivers can view patients")

    links = db.query(CaregiverPatientLinkModel).filter_by(caregiver_id=user.id).all()
    
    patients = []
    for link in links:
        patient_user = db.query(User).filter_by(id=link.patient_id).first()
        patient_profile = db.query(PatientProfile).filter_by(user_id=link.patient_id).first()
        if patient_user and patient_profile:
            patients.append({
                "user": patient_user,
                "profile": patient_profile
            })

    return patients


@router.delete("/{link_id}", response_model=dict)
def delete_link(
    link_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user)
):
    link = db.query(CaregiverPatientLinkModel).filter_by(id=link_id).first()

    if not link:
        raise HTTPException(status_code=404, detail="Link not found")

    if current_user_id not in [link.caregiver_id, link.patient_id]:
        raise HTTPException(status_code=403, detail="You are not part of this link")

    db.delete(link)
    db.commit()
    return {"message": "Link deleted successfully"}
