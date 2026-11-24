from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import List, Optional
from app.models.models import UserRole, EyeSide, ReferralStatus, ReferralTier

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    role: UserRole

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Patient Schemas
class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    phone: Optional[str] = None
    village: Optional[str] = None
    diabetes_duration: float = 0.0
    has_hypertension: bool = False
    has_prev_eye_issues: bool = False
    prev_eye_notes: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime
    created_by_id: int

    class Config:
        from_attributes = True

# Retinal Image Schemas
class RetinalImageBase(BaseModel):
    eye: EyeSide
    quality: str = "Good"
    gradability: str = "Gradable"
    focus_score: float = 1.0
    lighting_score: float = 1.0
    clarity_score: float = 1.0

class RetinalImageResponse(RetinalImageBase):
    id: int
    visit_id: int
    image_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

# Glaucoma Prediction Schemas
class GlaucomaPredictionResponse(BaseModel):
    id: int
    image_id: int
    prediction_class: str
    probability: float
    confidence: float
    risk_category: str
    heatmap_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Referral Schemas
class ReferralRecordBase(BaseModel):
    referred_to_facility: str
    notes: Optional[str] = None

class ReferralRecordCreate(ReferralRecordBase):
    patient_id: int
    visit_id: int

class ReferralRecordResponse(ReferralRecordBase):
    id: int
    patient_id: int
    visit_id: int
    status: ReferralStatus
    created_at: datetime

    class Config:
        from_attributes = True

# Generated Report Schemas
class GeneratedReportResponse(BaseModel):
    id: int
    visit_id: int
    report_path: str
    doctor_notes: Optional[str] = None
    is_signed: bool
    generated_at: datetime

    class Config:
        from_attributes = True

# Visit Schemas
class VisitBase(BaseModel):
    notes: Optional[str] = None
    capture_notes: Optional[str] = None

class VisitCreate(VisitBase):
    patient_id: int

class VisitResponse(VisitBase):
    id: int
    patient_id: int
    visit_date: datetime
    referral_needed: bool
    referral_reason: Optional[str] = None
    referral_tier: Optional[ReferralTier] = None
    advisory_note: Optional[str] = None
    created_by_id: int
    created_at: datetime
    images: List[RetinalImageResponse] = []
    reports: List[GeneratedReportResponse] = []

    class Config:
        from_attributes = True

class PatientDetailResponse(PatientResponse):
    visits: List[VisitResponse] = []

    class Config:
        from_attributes = True

# Dashboard Schemas
class DashboardStats(BaseModel):
    total_patients: int
    today_visits: int
    positive_cases: int
    negative_cases: int
    referral_counts: int
    monthly_statistics: dict
