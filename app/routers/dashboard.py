from datetime import datetime, timedelta, time
from collections import OrderedDict
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.models import Patient, Visit, ReferralRecord, User, GlaucomaPrediction, RetinalImage
from app.schemas.schemas import DashboardStats
from app.auth.auth_handler import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/stats", response_model=DashboardStats)
def get_dashboard_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Total Patients count
    total_patients = db.query(Patient).count()
    
    # Today's Visits count
    today_start = datetime.combine(datetime.utcnow().date(), time.min)
    today_end = datetime.combine(datetime.utcnow().date(), time.max)
    today_visits = db.query(Visit).filter(Visit.visit_date.between(today_start, today_end)).count()
    
    # Positive Cases: visits where glaucoma is detected in either eye
    # Or simply where referral_needed is True (since a referral is triggered by detection)
    positive_cases = db.query(Visit).filter(Visit.referral_needed == True).count()
    
    # Total completed screenings (visits with at least one prediction)
    completed_visits_count = db.query(Visit).join(Visit.images).join(RetinalImage.prediction).distinct().count()
    
    # Negative cases
    negative_cases = max(0, completed_visits_count - positive_cases)
    
    # Referral Counts
    referral_counts = db.query(ReferralRecord).count()
    
    # Monthly Statistics (Last 6 Months)
    monthly_stats = OrderedDict()
    current_date = datetime.utcnow()
    
    # Initialize last 6 months with 0
    for i in range(5, -1, -1):
        target_month_date = current_date - timedelta(days=i*30)
        month_name = target_month_date.strftime("%b")
        monthly_stats[month_name] = 0

    # Query database for visit counts grouped by month
    six_months_ago = current_date - timedelta(days=180)
    results = db.query(
        func.extract('month', Visit.visit_date).label('month'),
        func.count(Visit.id).label('count')
    ).filter(
        Visit.visit_date >= six_months_ago
    ).group_by(
        func.extract('month', Visit.visit_date)
    ).all()

    # Map month numbers to month abbreviations and update values
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    
    for r in results:
        m_num = int(r.month)
        m_name = month_map.get(m_num)
        if m_name in monthly_stats:
            monthly_stats[m_name] = r.count
            
    return {
        "total_patients": total_patients,
        "today_visits": today_visits,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "referral_counts": referral_counts,
        "monthly_statistics": dict(monthly_stats)
    }
