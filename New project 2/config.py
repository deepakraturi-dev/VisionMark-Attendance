from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)

    DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "database" / "smart_attendance.db"))
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DATABASE_PATH.resolve().as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DATASET_DIR = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset"))
    ENCODINGS_DIR = Path(os.getenv("ENCODINGS_DIR", BASE_DIR / "encodings"))
    ENCODINGS_PATH = Path(os.getenv("ENCODINGS_PATH", ENCODINGS_DIR / "faces.pkl"))
    REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))

    DEFAULT_CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    FACE_MATCH_TOLERANCE = float(os.getenv("FACE_MATCH_TOLERANCE", "0.48"))
    BEHAVIOR_LOG_INTERVAL_SECONDS = int(os.getenv("BEHAVIOR_LOG_INTERVAL_SECONDS", "5"))
    CAPTURE_IMAGE_COUNT = int(os.getenv("CAPTURE_IMAGE_COUNT", "30"))

    ADMIN_NAME = os.getenv("ADMIN_NAME", "System Administrator")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024


def ensure_project_directories(config: type[Config]) -> None:
    for path in (
        config.DATABASE_PATH.parent,
        config.DATASET_DIR,
        config.ENCODINGS_DIR,
        config.REPORTS_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
