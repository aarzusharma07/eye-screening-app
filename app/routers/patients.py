from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.models import Patient, User, ActivityLog
from app.schemas.schemas import PatientCreate, PatientResponse, PatientDetailResponse
from app.auth.auth_handler import get_current_user

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_patient = Patient(
        name=patient_in.name,
        age=patient_in.age,
        gender=patient_in.gender,
        phone=patient_in.phone,
        village=patient_in.village,
        diabetes_duration=patient_in.diabetes_duration,
        has_hypertension=patient_in.has_hypertension,
        has_prev_eye_issues=patient_in.has_prev_eye_issues,
        prev_eye_notes=patient_in.prev_eye_notes,
        created_by_id=current_user.id
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    # Log activity
    log = ActivityLog(
        user_id=current_user.id,
        action="Create Patient",
        details=f"Patient {db_patient.name} (ID: {db_patient.id}) registered by {current_user.username}"
    )
    db.add(log)
    db.commit()

    return db_patient

@router.get("", response_model=List[PatientResponse])
def get_patients(
    search: Optional[str] = Query(None, description="Search patients by name or village"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Patient)
    if search:
        query = query.filter(
            (Patient.name.ilike(f"%{search}%")) |
            (Patient.village.ilike(f"%{search}%")) |
            (Patient.phone.ilike(f"%{search}%"))
        )
    return query.order_by(Patient.created_at.desc()).all()

@router.get("/{patient_id}", response_model=PatientDetailResponse)
def get_patient_detail(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    return patient
