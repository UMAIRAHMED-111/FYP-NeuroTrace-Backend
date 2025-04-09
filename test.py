import uuid
from database import database
from models import users, patient_profiles, caregiver_profiles, caregiver_patient_links
from passlib.hash import bcrypt

async def seed():
    await database.connect()
    caregiver_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())
    
    await database.execute_many([
        users.insert().values(id=caregiver_id, email="care@example.com", password_hash=bcrypt.hash("pass123"), role="caregiver"),
        users.insert().values(id=patient_id, email="pat@example.com", password_hash=bcrypt.hash("pass123"), role="patient"),
        caregiver_profiles.insert().values(user_id=caregiver_id, full_name="Care Giver"),
        patient_profiles.insert().values(user_id=patient_id, full_name="Pat Ient"),
        caregiver_patient_links.insert().values(id=uuid.uuid4(), caregiver_id=caregiver_id, patient_id=patient_id)
    ])
    await database.disconnect()
