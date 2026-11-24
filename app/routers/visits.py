import os
from io import BytesIO
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image

from app.config import settings
from app.database.db import get_db
from app.models.models import (
    Visit, Patient, User, RetinalImage, GlaucomaPrediction,
    GeneratedReport, ReferralRecord, ActivityLog, EyeSide,
    ReferralStatus, ReferralTier
)
from app.schemas.schemas import VisitCreate, VisitResponse
from app.auth.auth_handler import get_current_user
from app.ml.ml_service import ml_service
from app.reports.report_service import ReportService

router = APIRouter(prefix="/visits", tags=["Visits"])


# -------------------------------------------------------------------
# Helper: compute multi-disease 4-tier referral triage + advisory
# -------------------------------------------------------------------
def _compute_referral_triage(predictions: list, disease_type: str = "both") -> tuple:
    """
    Returns (referral_needed: bool, referral_tier: ReferralTier,
             referral_reason: str, advisory_note: str)
    evaluating BOTH Glaucoma and Diabetic Retinopathy across all eye images.
    """
    glaucoma_max_prob = 0.0
    glaucoma_max_risk = "Low"
    dr_max_prob = 0.0
    dr_max_risk = "Low"
    
    glaucoma_reasons = []
    dr_reasons = []

    for pred in predictions:
        eye_label = pred.image.eye.value
        
        # 1. Glaucoma evaluation
        g_prob = pred.probability
        g_risk = pred.risk_category
        if g_prob > glaucoma_max_prob:
            glaucoma_max_prob = g_prob
            glaucoma_max_risk = g_risk
        if pred.prediction_class == "glaucoma" or g_prob >= 0.35:
            glaucoma_reasons.append(
                f"{eye_label} Eye: {g_risk} Risk Glaucoma ({g_prob * 100:.1f}%)"
            )

        # 2. Diabetic Retinopathy evaluation
        dr_prob = pred.dr_probability if pred.dr_probability is not None else 0.0
        dr_risk = pred.dr_risk_category if pred.dr_risk_category else "Low"
        dr_cls = pred.dr_prediction_class or "No DR"
        if dr_prob > dr_max_prob:
            dr_max_prob = dr_prob
            dr_max_risk = dr_risk
        if dr_cls != "No DR" or dr_prob >= 0.35:
            dr_reasons.append(
                f"{eye_label} Eye: {dr_risk} Risk DR [{dr_cls}] ({dr_prob * 100:.1f}%)"
            )

    all_reasons = glaucoma_reasons + dr_reasons

    # 4-tier combined triage logic
    is_urgent = (glaucoma_max_prob >= 0.70 or glaucoma_max_risk == "High" or
                 dr_max_prob >= 0.70 or dr_max_risk == "High")
    is_refer = (glaucoma_max_prob >= 0.50 or glaucoma_max_risk == "Medium" or
                dr_max_prob >= 0.50 or dr_max_risk == "Medium")
    is_monitor = (glaucoma_max_prob >= 0.25 or dr_max_prob >= 0.25 or len(all_reasons) > 0)

    if is_urgent:
        tier = ReferralTier.URGENT
        referral_needed = True
        referral_reason = "🔴 URGENT: High-risk ocular indicators detected. " + ("; ".join(all_reasons) if all_reasons else "Immediate ophthalmologist evaluation needed.")
        advisory = (
            "URGENT REFERRAL REQUIRED: High-risk indicators identified for Glaucoma and/or Severe Diabetic Retinopathy. "
            "Refer patient to an ophthalmology specialist within 24–48 hours for complete dilated fundus exam, "
            "tonometry (IOP check), and retinal assessment. Delay in clinical evaluation may cause irreversible vision loss."
        )
    elif is_refer:
        tier = ReferralTier.REFER
        referral_needed = True
        referral_reason = "🟠 REFERRAL RECOMMENDED: Moderate risk indicators detected. " + ("; ".join(all_reasons) if all_reasons else "Follow-up required.")
        advisory = (
            "REFERRAL RECOMMENDED: AI screening detected moderate risk indicators for Glaucoma and/or Diabetic Retinopathy. "
            "Refer patient to a district eye clinic within 1–2 weeks for detailed evaluation. "
            "Advise patient on glycemic control (HbA1c target), blood pressure maintenance, and reporting any acute vision changes."
        )
    elif is_monitor:
        tier = ReferralTier.MONITOR
        referral_needed = False
        referral_reason = "🟡 MONITOR: Borderline findings detected. " + ("; ".join(all_reasons) if all_reasons else "Routine follow-up advised in 6 months.")
        advisory = (
            "MONITOR: Borderline or mild findings detected that do not require immediate emergency referral. "
            "Schedule a routine follow-up screening in 6 months. Maintain systemic health management (diabetes & hypertension)."
        )
    else:
        tier = ReferralTier.REASSURE
        referral_needed = False
        referral_reason = "🟢 REASSURE: Normal findings across both Glaucoma & Diabetic Retinopathy screening."
        advisory = (
            "REASSURE: No significant Glaucoma or Diabetic Retinopathy indicators detected. "
            "Reassure the patient and recommend routine annual preventive retinal screening in 12 months."
        )

    return referral_needed, tier, referral_reason, advisory


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@router.post("", response_model=VisitResponse, status_code=status.HTTP_201_CREATED)
def create_visit(
    visit_in: VisitCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == visit_in.patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )

    db_visit = Visit(
        patient_id=visit_in.patient_id,
        notes=visit_in.notes,
        capture_notes=visit_in.capture_notes,
        created_by_id=current_user.id
    )
    db.add(db_visit)
    db.commit()
    db.refresh(db_visit)

    # Log activity
    log = ActivityLog(
        user_id=current_user.id,
        action="Create Visit",
        details=f"Visit {db_visit.id} created for patient {patient.name} (ID: {patient.id}) by {current_user.username}"
    )
    db.add(log)
    db.commit()

    return db_visit


@router.post("/{visit_id}/upload-images")
def upload_images(
    visit_id: int,
    left_eye: UploadFile = File(...),
    right_eye: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fetch visit
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visit not found"
        )

    patient_id = visit.patient_id
    visit_dir = os.path.join(settings.UPLOAD_DIR, f"patient_{patient_id}", f"visit_{visit_id}")
    os.makedirs(visit_dir, exist_ok=True)

    saved_images = []

    for file, side in [(left_eye, EyeSide.LEFT), (right_eye, EyeSide.RIGHT)]:
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext and file.content_type in {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/avif"}:
            ext = ".png" if file.content_type == "image/png" else ".webp" if file.content_type == "image/webp" else ".avif" if file.content_type == "image/avif" else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format {ext} for {side.value} eye. Only JPG, PNG, WebP, and AVIF are supported."
            )

        file_bytes = file.file.read()
        if ext == ".avif":
            ext = ".jpg"
            converted = BytesIO()
            Image.open(BytesIO(file_bytes)).convert("RGB").save(converted, format="JPEG", quality=95)
            file_bytes = converted.getvalue()

        filename = f"{side.value.lower()}{ext}"
        save_path = os.path.join(visit_dir, filename)

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # Perform image quality checks
        quality_info = ml_service.analyze_image_quality(save_path)

        # Gradability: ungradable if quality is Poor AND focus very low
        gradability = "Ungradable" if (quality_info["quality"] == "Poor" and quality_info["focus"] < 0.3) else "Gradable"

        # Delete existing image record if re-uploaded
        existing_img = db.query(RetinalImage).filter(
            RetinalImage.visit_id == visit_id,
            RetinalImage.eye == side
        ).first()
        if existing_img:
            db.query(GlaucomaPrediction).filter(GlaucomaPrediction.image_id == existing_img.id).delete()
            db.delete(existing_img)

        db_image = RetinalImage(
            visit_id=visit_id,
            eye=side,
            image_path=save_path,
            quality=quality_info["quality"],
            gradability=gradability,
            focus_score=quality_info["focus"],
            lighting_score=quality_info["lighting"],
            clarity_score=quality_info["clarity"]
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        saved_images.append(db_image)

    # Log activity
    log = ActivityLog(
        user_id=current_user.id,
        action="Upload Images",
        details=f"Uploaded retinal images for visit {visit_id} (Patient ID: {patient_id}) by {current_user.username}"
    )
    db.add(log)
    db.commit()

    return {
        "message": "Images uploaded and quality verified successfully",
        "images": [
            {
                "id": img.id,
                "eye": img.eye.value,
                "quality": img.quality,
                "gradability": img.gradability,
                "focus": img.focus_score,
                "lighting": img.lighting_score,
                "clarity": img.clarity_score,
                "recapture_needed": img.quality == "Poor"
            } for img in saved_images
        ]
    }


@router.post("/{visit_id}/analyze")
def analyze_visit(
    visit_id: int,
    doctor_notes: Optional[str] = Form(None),
    capture_notes: Optional[str] = Form(None),
    disease_type: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fetch visit
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visit not found")

    if not visit.images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images uploaded for this visit. Upload images before running prediction."
        )

    # Update notes if provided
    if doctor_notes:
        visit.notes = doctor_notes
    if capture_notes:
        visit.capture_notes = capture_notes
    db.commit()

    # Dir for heatmaps
    patient = visit.patient
    patient_id = visit.patient_id
    heatmap_dir = os.path.join(settings.UPLOAD_DIR, f"patient_{patient_id}", f"visit_{visit_id}")
    os.makedirs(heatmap_dir, exist_ok=True)

    predictions = []

    for img in visit.images:
        # Run BOTH Glaucoma and DR predictions on the retinal image
        g_pred_data, g_heatmap_path = ml_service.predict_glaucoma(img.image_path, heatmap_dir)
        dr_pred_data, dr_heatmap_path = ml_service.predict_dr(img.image_path, heatmap_dir, patient.diabetes_duration)

        # Delete existing prediction if any
        db.query(GlaucomaPrediction).filter(GlaucomaPrediction.image_id == img.id).delete()

        db_pred = GlaucomaPrediction(
            image_id=img.id,
            # Glaucoma
            prediction_class=g_pred_data["prediction_class"],
            probability=g_pred_data["probability"],
            confidence=g_pred_data["confidence"],
            risk_category=g_pred_data["risk_category"],
            heatmap_path=g_heatmap_path,
            # Diabetic Retinopathy
            dr_prediction_class=dr_pred_data["prediction_class"],
            dr_probability=dr_pred_data["probability"],
            dr_confidence=dr_pred_data["confidence"],
            dr_risk_category=dr_pred_data["risk_category"],
            dr_heatmap_path=dr_heatmap_path
        )
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)
        predictions.append(db_pred)

    # Multi-disease 4-tier referral triage
    referral_needed, tier, referral_reason, advisory = _compute_referral_triage(predictions, "both")

    visit.referral_needed = referral_needed
    visit.referral_tier = tier
    visit.referral_reason = referral_reason
    visit.advisory_note = advisory

    if referral_needed:
        # Create or update referral record
        existing_ref = db.query(ReferralRecord).filter(ReferralRecord.visit_id == visit_id).first()
        if not existing_ref:
            facility = (
                "Emergency Ophthalmology — District Hospital"
                if tier == ReferralTier.URGENT
                else "District Eye Clinic / Ophthalmology Specialist"
            )
            db_ref = ReferralRecord(
                patient_id=patient_id,
                visit_id=visit_id,
                referred_to_facility=facility,
                status=ReferralStatus.PENDING,
                notes=f"AI Multi-Screening ({tier.value}): {referral_reason}"
            )
            db.add(db_ref)
    else:
        db.query(ReferralRecord).filter(ReferralRecord.visit_id == visit_id).delete()

    db.commit()
    db.refresh(visit)

    # Generate comprehensive clinical PDF report
    report_path = ReportService.generate_pdf_report(visit, db)

    db.query(GeneratedReport).filter(GeneratedReport.visit_id == visit_id).delete()

    db_report = GeneratedReport(
        visit_id=visit_id,
        report_path=report_path,
        doctor_notes=visit.notes,
        is_signed=True
    )
    db.add(db_report)
    db.commit()

    # Log activity
    log = ActivityLog(
        user_id=current_user.id,
        action="Run Prediction",
        details=f"Multi-disease (Glaucoma + DR) screening executed for visit {visit_id}. Triage: {tier.value}."
    )
    db.add(log)
    db.commit()

    return {
        "visit_id": visit.id,
        "referral_needed": visit.referral_needed,
        "referral_tier": visit.referral_tier.value if visit.referral_tier else None,
        "referral_reason": visit.referral_reason,
        "advisory_note": visit.advisory_note,
        "predictions": [
            {
                "eye": pred.image.eye.value,
                # Glaucoma
                "class": pred.prediction_class,
                "probability": pred.probability,
                "confidence": pred.confidence,
                "risk_category": pred.risk_category,
                "glaucoma_class": pred.prediction_class,
                "glaucoma_probability": pred.probability,
                "glaucoma_confidence": pred.confidence,
                "glaucoma_risk": pred.risk_category,
                # Diabetic Retinopathy
                "dr_class": pred.dr_prediction_class or "No DR",
                "dr_probability": pred.dr_probability if pred.dr_probability is not None else 0.0,
                "dr_confidence": pred.dr_confidence if pred.dr_confidence is not None else 0.0,
                "dr_risk": pred.dr_risk_category or "Low",
                # Quality
                "quality": pred.image.quality,
                "gradability": pred.image.gradability,
            } for pred in predictions
        ],
        "report": {
            "id": db_report.id,
            "report_path": db_report.report_path,
            "generated_at": db_report.generated_at
        }
    }


@router.get("/{visit_id}", response_model=VisitResponse)
def get_visit_details(
    visit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visit not found")
    return visit


@router.get("/{visit_id}/report")
def download_report(
    visit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    report = db.query(GeneratedReport).filter(GeneratedReport.visit_id == visit_id).first()
    if not report or not os.path.exists(report.report_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clinical PDF report not found. Run prediction/analyze first."
        )
    return FileResponse(
        report.report_path,
        media_type="application/pdf",
        filename=f"eye_screening_report_visit_{visit_id}.pdf"
    )


@router.get("/{visit_id}/heatmap/{eye}")
def get_heatmap(
    visit_id: int,
    eye: str,
    disease: str = "glaucoma",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        eye_side = EyeSide(eye.capitalize())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Eye must be 'Left' or 'Right'")

    img = db.query(RetinalImage).filter(
        RetinalImage.visit_id == visit_id,
        RetinalImage.eye == eye_side
    ).first()

    if not img or not img.prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction for {eye} eye not found. Please upload images and run prediction."
        )

    if disease.lower() == "dr":
        path = img.prediction.dr_heatmap_path or img.prediction.heatmap_path
    else:
        path = img.prediction.heatmap_path

    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Heatmap ({disease}) for {eye} eye not found."
        )

    return FileResponse(path, media_type="image/jpeg")


@router.get("/{visit_id}/original/{eye}")
def get_original_image(
    visit_id: int,
    eye: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        eye_side = EyeSide(eye.capitalize())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Eye must be 'Left' or 'Right'")

    img = db.query(RetinalImage).filter(
        RetinalImage.visit_id == visit_id,
        RetinalImage.eye == eye_side
    ).first()

    if not img or not os.path.exists(img.image_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original retinal scan for {eye} eye not found."
        )

    ext = os.path.splitext(img.image_path)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(img.image_path, media_type=media_type)
