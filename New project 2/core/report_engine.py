from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from database.models import Attendance, BehaviorLog, Student


class ReportEngine:
    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def daily_attendance_csv(self, target_date: date) -> Path:
        path = self.reports_dir / f"daily_attendance_{target_date.isoformat()}.csv"
        records = (
            Attendance.query.join(Student)
            .filter(Attendance.date == target_date)
            .order_by(Attendance.time.asc())
            .all()
        )
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["student_id", "name", "roll_no", "email", "date", "time", "status"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "student_id": record.student_id,
                        "name": record.student.name,
                        "roll_no": record.student.roll_no,
                        "email": record.student.email,
                        "date": record.date.isoformat(),
                        "time": record.time.isoformat(timespec="seconds"),
                        "status": record.status,
                    }
                )
        return path

    def monthly_attendance_csv(self, year: int, month: int, summary: list[dict]) -> Path:
        path = self.reports_dir / f"monthly_attendance_{year}_{month:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["student_id", "name", "roll_no", "email", "present_days"],
            )
            writer.writeheader()
            writer.writerows(summary)
        return path

    def behavior_csv(self, start_at: datetime, end_at: datetime, summary: list[dict]) -> Path:
        path = self.reports_dir / f"behavior_{start_at.date().isoformat()}_{end_at.date().isoformat()}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "student_id",
                "name",
                "roll_no",
                "total_logs",
                "average_attention",
                "attentive_percent",
                "distracted_percent",
                "sleepy_percent",
                "unknown_percent",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def daily_attendance_pdf(self, target_date: date) -> Path:
        rows = (
            Attendance.query.join(Student)
            .filter(Attendance.date == target_date)
            .order_by(Attendance.time.asc())
            .all()
        )
        table_rows = [["Roll No", "Name", "Email", "Time", "Status"]]
        for record in rows:
            table_rows.append(
                [
                    record.student.roll_no,
                    record.student.name,
                    record.student.email,
                    record.time.isoformat(timespec="seconds"),
                    record.status.title(),
                ]
            )
        return self._build_pdf(
            filename=f"daily_attendance_{target_date.isoformat()}.pdf",
            title=f"Daily Attendance Report - {target_date.isoformat()}",
            table_rows=table_rows,
        )

    def behavior_pdf(self, start_at: datetime, end_at: datetime, summary: list[dict]) -> Path:
        table_rows = [["Roll No", "Name", "Avg Score", "Attentive %", "Distracted %", "Sleepy %"]]
        for row in summary:
            table_rows.append(
                [
                    row["roll_no"],
                    row["name"],
                    row["average_attention"],
                    row["attentive_percent"],
                    row["distracted_percent"],
                    row["sleepy_percent"],
                ]
            )
        return self._build_pdf(
            filename=f"behavior_{start_at.date().isoformat()}_{end_at.date().isoformat()}.pdf",
            title=f"Behavior Report - {start_at.date().isoformat()} to {end_at.date().isoformat()}",
            table_rows=table_rows,
        )

    def _build_pdf(self, filename: str, title: str, table_rows: list[list]) -> Path:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as exc:  # pragma: no cover - dependency listed in requirements
            raise RuntimeError("PDF export requires reportlab. Install dependencies first.") from exc

        path = self.reports_dir / filename
        document = SimpleDocTemplate(str(path), pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph(title, styles["Title"]), Spacer(1, 16)]
        table = Table(table_rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(table)
        document.build(elements)
        return path
