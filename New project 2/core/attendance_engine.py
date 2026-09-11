from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from database.db import db
from database.models import Attendance, BehaviorLog, Student


class AttendanceEngine:
    def __init__(self, reports_dir: str | Path, behavior_log_interval_seconds: int = 5) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.behavior_log_interval_seconds = behavior_log_interval_seconds
        self._behavior_last_logged: dict[int, datetime] = {}
        self._lock = Lock()

    def _attendance_csv_path(self, target_date: date) -> Path:
        return self.reports_dir / f"attendance_{target_date.isoformat()}.csv"

    def _append_attendance_csv(self, record: Attendance) -> None:
        path = self._attendance_csv_path(record.date)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["student_id", "name", "roll_no", "date", "time", "status", "source_camera"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "student_id": record.student_id,
                    "name": record.student.name,
                    "roll_no": record.student.roll_no,
                    "date": record.date.isoformat(),
                    "time": record.time.isoformat(timespec="seconds"),
                    "status": record.status,
                    "source_camera": record.source_camera or "",
                }
            )

    def mark_attendance(
        self,
        student: Student,
        source_camera: str | int | None = None,
        attended_at: datetime | None = None,
    ) -> tuple[Attendance, bool]:
        attended_at = attended_at or datetime.now()
        target_date = attended_at.date()

        with self._lock:
            existing = Attendance.query.filter_by(student_id=student.id, date=target_date).first()
            if existing:
                return existing, False

            record = Attendance(
                student_id=student.id,
                date=target_date,
                time=attended_at.time().replace(microsecond=0),
                status="present",
                source_camera=str(source_camera) if source_camera is not None else None,
            )
            db.session.add(record)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                existing = Attendance.query.filter_by(student_id=student.id, date=target_date).first()
                if existing:
                    return existing, False
                raise

            record.student = student
            self._append_attendance_csv(record)
            return record, True

    def log_behavior(self, student_id: int, behavior_result, force: bool = False) -> BehaviorLog | None:
        now = datetime.now()
        last_logged = self._behavior_last_logged.get(student_id)
        if (
            not force
            and last_logged is not None
            and (now - last_logged).total_seconds() < self.behavior_log_interval_seconds
        ):
            return None

        payload = behavior_result.to_dict() if hasattr(behavior_result, "to_dict") else dict(behavior_result)
        log = BehaviorLog(
            student_id=student_id,
            timestamp=now,
            attention_score=int(payload["attention_score"]),
            status=payload["status"],
            ear=payload.get("ear"),
            head_yaw=payload.get("head_yaw"),
            head_pitch=payload.get("head_pitch"),
            head_roll=payload.get("head_roll"),
            alert=payload.get("alert"),
        )
        db.session.add(log)
        db.session.commit()
        self._behavior_last_logged[student_id] = now
        return log

    def attendance_for_date(self, target_date: date) -> list[Attendance]:
        return (
            Attendance.query.join(Student)
            .filter(Attendance.date == target_date)
            .order_by(Attendance.time.asc())
            .all()
        )

    def monthly_summary(self, year: int, month: int) -> list[dict]:
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, month + 1, 1)

        rows = (
            db.session.query(Student, func.count(Attendance.id).label("present_days"))
            .outerjoin(
                Attendance,
                (Attendance.student_id == Student.id)
                & (Attendance.date >= start)
                & (Attendance.date < end),
            )
            .filter(Student.active.is_(True))
            .group_by(Student.id)
            .order_by(Student.roll_no.asc())
            .all()
        )

        return [
            {
                "student_id": student.id,
                "name": student.name,
                "roll_no": student.roll_no,
                "email": student.email,
                "present_days": int(present_days or 0),
            }
            for student, present_days in rows
        ]

    def behavior_summary(
        self,
        start_at: datetime,
        end_at: datetime,
        student_id: int | None = None,
    ) -> list[dict]:
        query = BehaviorLog.query.join(Student).filter(
            BehaviorLog.timestamp >= start_at,
            BehaviorLog.timestamp < end_at,
        )
        if student_id:
            query = query.filter(BehaviorLog.student_id == student_id)

        logs = query.order_by(BehaviorLog.timestamp.asc()).all()
        grouped: dict[int, dict] = {}
        for log in logs:
            bucket = grouped.setdefault(
                log.student_id,
                {
                    "student_id": log.student_id,
                    "name": log.student.name,
                    "roll_no": log.student.roll_no,
                    "total_logs": 0,
                    "average_attention": 0.0,
                    "attentive": 0,
                    "distracted": 0,
                    "sleepy": 0,
                    "unknown": 0,
                },
            )
            bucket["total_logs"] += 1
            bucket["average_attention"] += log.attention_score
            bucket[log.status if log.status in bucket else "unknown"] += 1

        for bucket in grouped.values():
            total = bucket["total_logs"] or 1
            bucket["average_attention"] = round(bucket["average_attention"] / total, 2)
            for status in ("attentive", "distracted", "sleepy", "unknown"):
                bucket[f"{status}_percent"] = round((bucket[status] / total) * 100, 2)

        return list(grouped.values())

    @staticmethod
    def day_bounds(target_date: date) -> tuple[datetime, datetime]:
        start = datetime.combine(target_date, datetime.min.time())
        return start, start + timedelta(days=1)
