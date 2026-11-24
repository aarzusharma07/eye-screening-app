import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load the .env from the project root so env vars
# are available before pydantic_settings reads them.
load_dotenv(BASE_DIR / ".env", override=True)


def resolve_path(value: Optional[str], default: str) -> str:
    candidate = value if value is not None else default
    path = Path(candidate)
    if path.is_absolute():
        return str(path)
    return str((BASE_DIR / path).resolve())


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str = "glaucoma_super_secret_jwt_key_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Storage settings
    UPLOAD_DIR: str = resolve_path(os.getenv("UPLOAD_DIR"), "./data/images")
    REPORT_DIR: str = resolve_path(os.getenv("REPORT_DIR"), "./data/reports")

    # Model settings
    MODEL_PATH: str = resolve_path(os.getenv("MODEL_PATH"), "./app/ml/best_glaucoma_model_refined.pth")
    DR_MODEL_PATH: str = resolve_path(os.getenv("DR_MODEL_PATH"), "./app/ml/best_dr_model.pth")

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Ensure directories exist locally
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.REPORT_DIR).mkdir(parents=True, exist_ok=True)
