from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database.db import engine, Base, SessionLocal
from app.models.models import User, UserRole
from app.auth.auth_handler import get_password_hash
from app.routers import auth, patients, visits, dashboard

app = FastAPI(
    title="Glaucoma Eye Screening API Backend",
    description="Production-ready clinical AI eye screening backend with PyTorch classification, OpenCV quality assessment, and ReportLab PDF reporting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Set up CORS middleware to allow Flutter app requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from sqlalchemy import text

# Startup DB initialization & default user seeding
@app.on_event("startup")
def on_startup():
    # Create tables automatically in development
    Base.metadata.create_all(bind=engine)
    
    # Safe migration for new DR columns
    try:
        with engine.connect() as conn:
            for col_sql in [
                "ALTER TABLE glaucoma_predictions ADD COLUMN dr_prediction_class VARCHAR(50)",
                "ALTER TABLE glaucoma_predictions ADD COLUMN dr_probability FLOAT",
                "ALTER TABLE glaucoma_predictions ADD COLUMN dr_confidence FLOAT",
                "ALTER TABLE glaucoma_predictions ADD COLUMN dr_risk_category VARCHAR(20)",
                "ALTER TABLE glaucoma_predictions ADD COLUMN dr_heatmap_path VARCHAR(255)"
            ]:
                try:
                    conn.execute(text(col_sql))
                    conn.commit()
                except Exception:
                    pass
    except Exception as e:
        print(f"Migration note: {e}")
    
    # Seed default users
    db = SessionLocal()
    try:
        # Check if users exist, if not seed them
        user_count = db.query(User).count()
        if user_count == 0:
            print("No users found. Seeding default roles and credentials...")
            
            users_to_seed = [
                {
                    "username": "admin",
                    "email": "admin@eyescreening.org",
                    "password": "password",
                    "full_name": "Administrator System",
                    "role": UserRole.ADMIN
                },
                {
                    "username": "doctor",
                    "email": "doctor@eyescreening.org",
                    "password": "password",
                    "full_name": "Dr. Sarah Jenkins",
                    "role": UserRole.DOCTOR
                },
                {
                    "username": "nurse",
                    "email": "nurse@eyescreening.org",
                    "password": "password",
                    "full_name": "Clara Barton, RN",
                    "role": UserRole.NURSE
                }
            ]
            
            for u in users_to_seed:
                hashed = get_password_hash(u["password"])
                db_user = User(
                    username=u["username"],
                    email=u["email"],
                    hashed_password=hashed,
                    full_name=u["full_name"],
                    role=u["role"]
                )
                db.add(db_user)
            db.commit()
            print("Default users seeded successfully! Logins available: doctor/password, nurse/password, admin/password.")
    except Exception as e:
        print(f"Error seeding default users: {e}")
    finally:
        db.close()

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(patients.router, prefix="/api")
app.include_router(visits.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "Glaucoma AI Eye Screening API Backend",
        "version": "1.0.0",
        "cuda_available": False # Default false, will check torch inside
    }
