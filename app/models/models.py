import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Enum as SQLEnum, Text
from sqlalchemy.orm import relationship
from app.database.db import Base

class UserRole(str, enum.Enum):
    ADMIN = "Admin"
    DOCTOR = "Doctor"
    NURSE = "Nurse"

class EyeSide(str, enum.Enum):
    LEFT = "Left"
    RIGHT = "Right"

class ReferralStatus(str, enum.Enum):
    PENDING = "Pending"
    COMPLETED = "Completed"

class ReferralTier(str, enum.Enum):
    REASSURE = "Reassure"
    MONITOR = "Monitor"
    REFER = "Refer"
    URGENT = "Urgent Referral"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.NURSE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patients_created = relationship("Patient", back_populates="created_by")
    visits_created = relationship("Visit", back_populates="created_by")
    activity_logs = relationship("ActivityLog", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    phone = Column(String(20), nullable=True)
    village = Column(String(100), nullable=True)
    diabetes_duration = Column(Float, default=0.0)  # in years
    has_hypertension = Column(Boolean, default=False)
    has_prev_eye_issues = Column(Boolean, default=False)
    prev_eye_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    created_by = relationship("User", back_populates="patients_created")
    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")
    referrals = relationship("ReferralRecord", back_populates="patient", cascade="all, delete-orphan")


class Visit(Base):
    __tablename__ = "patient_visits"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    capture_notes = Column(Text, nullable=True)      # nurse capture notes (e.g. poor cooperation)
    referral_needed = Column(Boolean, default=False)
    referral_reason = Column(String(255), nullable=True)
    referral_tier = Column(SQLEnum(ReferralTier), nullable=True)   # 4-tier triage
    advisory_note = Column(Text, nullable=True)       # grounded clinical advisory
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="visits")
    created_by = relationship("User", back_populates="visits_created")
    images = relationship("RetinalImage", back_populates="visit", cascade="all, delete-orphan")
    reports = relationship("GeneratedReport", back_populates="visit", cascade="all, delete-orphan")
    referral_records = relationship("ReferralRecord", back_populates="visit", cascade="all, delete-orphan")


class RetinalImage(Base):
    __tablename__ = "retinal_images"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=False)
    eye = Column(SQLEnum(EyeSide), nullable=False)
    image_path = Column(String(255), nullable=False)
    quality = Column(String(50), default="Good")   # Good, Poor, Ungradable
    gradability = Column(String(50), default="Gradable")  # Gradable, Ungradable
    focus_score = Column(Float, default=1.0)
    lighting_score = Column(Float, default=1.0)
    clarity_score = Column(Float, default=1.0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    visit = relationship("Visit", back_populates="images")
    prediction = relationship("GlaucomaPrediction", back_populates="image", uselist=False, cascade="all, delete-orphan")


class GlaucomaPrediction(Base):
    __tablename__ = "glaucoma_predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("retinal_images.id"), nullable=False)
    
    # Glaucoma findings
    prediction_class = Column(String(50), nullable=False)   # glaucoma, normal
    probability = Column(Float, nullable=False)             # probability of glaucoma
    confidence = Column(Float, nullable=False)              # model confidence
    risk_category = Column(String(20), nullable=False)      # Low, Medium, High
    heatmap_path = Column(String(255), nullable=True)       # path to Glaucoma Grad-CAM output
    
    # Diabetic Retinopathy findings
    dr_prediction_class = Column(String(50), nullable=True) # No DR, Mild, Moderate, Severe, Proliferative
    dr_probability = Column(Float, nullable=True)           # probability of DR
    dr_confidence = Column(Float, nullable=True)            # DR confidence
    dr_risk_category = Column(String(20), nullable=True)    # Low, Medium, High
    dr_heatmap_path = Column(String(255), nullable=True)    # path to DR Grad-CAM output
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    image = relationship("RetinalImage", back_populates="prediction")


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=False)
    report_path = Column(String(255), nullable=False)
    doctor_notes = Column(Text, nullable=True)
    is_signed = Column(Boolean, default=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    visit = relationship("Visit", back_populates="reports")


class ReferralRecord(Base):
    __tablename__ = "referral_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=False)
    referred_to_facility = Column(String(150), nullable=False)
    status = Column(SQLEnum(ReferralStatus), default=ReferralStatus.PENDING)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="referrals")
    visit = relationship("Visit", back_populates="referral_records")


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="activity_logs")
