from __future__ import annotations

from datetime import date, datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from database.db import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="teacher")
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    @property
    def is_active(self) -> bool:
        return bool(self.active)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
        }


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    dataset_path = db.Column(db.String(500), nullable=True)
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    attendance_records = db.relationship(
        "Attendance",
        back_populates="student",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    behavior_logs = db.relationship(
        "BehaviorLog",
        back_populates="student",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "roll_no": self.roll_no,
            "email": self.email,
            "dataset_path": self.dataset_path,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
        }


class Attendance(db.Model):
    __tablename__ = "attendance"
    __table_args__ = (
        db.UniqueConstraint("student_id", "date", name="uq_attendance_student_date"),
        db.Index("ix_attendance_date_student", "date", "student_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="present")
    source_camera = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    student = db.relationship("Student", back_populates="attendance_records")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "roll_no": self.student.roll_no if self.student else None,
            "date": self.date.isoformat(),
            "time": self.time.isoformat(timespec="seconds"),
            "status": self.status,
            "source_camera": self.source_camera,
            "created_at": self.created_at.isoformat(),
        }


class BehaviorLog(db.Model):
    __tablename__ = "behavior_logs"
    __table_args__ = (
        db.Index("ix_behavior_student_timestamp", "student_id", "timestamp"),
        db.Index("ix_behavior_timestamp_status", "timestamp", "status"),
    )

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    attention_score = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(30), nullable=False)
    ear = db.Column(db.Float, nullable=True)
    head_yaw = db.Column(db.Float, nullable=True)
    head_pitch = db.Column(db.Float, nullable=True)
    head_roll = db.Column(db.Float, nullable=True)
    alert = db.Column(db.String(255), nullable=True)

    student = db.relationship("Student", back_populates="behavior_logs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "roll_no": self.student.roll_no if self.student else None,
            "timestamp": self.timestamp.isoformat(),
            "attention_score": self.attention_score,
            "status": self.status,
            "ear": self.ear,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "head_roll": self.head_roll,
            "alert": self.alert,
        }
