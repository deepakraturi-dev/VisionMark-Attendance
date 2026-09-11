from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

import jwt
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import func

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None

from config import Config, ensure_project_directories
from core.attendance_engine import AttendanceEngine
from core.behavior_engine import BehaviorEngine, BehaviorEngineError
from core.face_engine import FaceEngine, FaceEngineError
from core.report_engine import ReportEngine
from database.db import db, init_database, init_extensions, login_manager
from database.models import Attendance, BehaviorLog, Student, User


def parse_date(value: str | None, default: date | None = None) -> date:
    if not value:
        if default is None:
            raise ValueError("Date is required.")
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_month(value: str | None) -> tuple[int, int]:
    if not value:
        today = date.today()
        return today.year, today.month
    parsed = datetime.strptime(value, "%Y-%m")
    return parsed.year, parsed.month


def create_app() -> Flask:
    ensure_project_directories(Config)

    app = Flask(__name__)
    app.config.from_object(Config)

    init_extensions(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    init_database(app)

    face_engine = FaceEngine(
        dataset_dir=app.config["DATASET_DIR"],
        encodings_path=app.config["ENCODINGS_PATH"],
        tolerance=app.config["FACE_MATCH_TOLERANCE"],
    )
    behavior_engine = BehaviorEngine()
    attendance_engine = AttendanceEngine(
        reports_dir=app.config["REPORTS_DIR"],
        behavior_log_interval_seconds=app.config["BEHAVIOR_LOG_INTERVAL_SECONDS"],
    )
    report_engine = ReportEngine(app.config["REPORTS_DIR"])

    app.extensions["smart_attendance"] = {
        "face": face_engine,
        "behavior": behavior_engine,
        "attendance": attendance_engine,
        "reports": report_engine,
    }

    @app.context_processor
    def inject_globals():
        return {"today": date.today(), "current_year": date.today().year}

    def create_token(user: User) -> str:
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(hours=8),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, app.config["JWT_SECRET_KEY"], algorithm="HS256")

    def jwt_required(handler):
        @wraps(handler)
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing bearer token."}), 401
            token = auth_header.split(" ", 1)[1].strip()
            try:
                payload = jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
            except jwt.PyJWTError as exc:
                return jsonify({"error": f"Invalid token: {exc}"}), 401

            user = db.session.get(User, int(payload["sub"]))
            if not user or not user.active:
                return jsonify({"error": "User is not active."}), 401
            g.jwt_user = user
            return handler(*args, **kwargs)

        return wrapper

    def camera_index_from_request() -> int:
        try:
            return int(request.args.get("camera", app.config["DEFAULT_CAMERA_INDEX"]))
        except (TypeError, ValueError):
            return app.config["DEFAULT_CAMERA_INDEX"]

    def capture_count_from_payload(payload) -> int:
        try:
            count = int(payload.get("count", app.config["CAPTURE_IMAGE_COUNT"]))
        except (TypeError, ValueError):
            count = app.config["CAPTURE_IMAGE_COUNT"]
        return max(20, min(50, count))

    def camera_index_from_payload(payload) -> int:
        try:
            return int(payload.get("camera", app.config["DEFAULT_CAMERA_INDEX"]))
        except (TypeError, ValueError):
            return app.config["DEFAULT_CAMERA_INDEX"]

    def encode_frame(frame):
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buffer.tobytes()

    def message_frame(message: str):
        if cv2 is None or np is None:
            return None
        frame = np.zeros((480, 820, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            message[:80],
            (28, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def mjpeg(frame_bytes: bytes):
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

    def preview_stream(camera_index: int):
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            frame = message_frame(f"Unable to open camera {camera_index}")
            if frame is not None:
                yield mjpeg(encode_frame(frame))
            return

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    continue
                frame_bytes = encode_frame(frame)
                if frame_bytes:
                    yield mjpeg(frame_bytes)
        finally:
            camera.release()

    def attendance_stream(camera_index: int):
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            frame = message_frame(f"Unable to open camera {camera_index}")
            if frame is not None:
                yield mjpeg(encode_frame(frame))
            return

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    continue

                try:
                    recognitions = face_engine.recognize_frame(frame)
                    face_engine.draw_recognition_overlay(frame, recognitions)
                except FaceEngineError as exc:
                    recognitions = []
                    cv2.putText(
                        frame,
                        str(exc)[:110],
                        (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                for face in recognitions:
                    if not face.get("recognized"):
                        continue
                    student = db.session.get(Student, face["student_id"])
                    if not student or not student.active:
                        continue

                    attendance_engine.mark_attendance(student, source_camera=camera_index)

                    try:
                        behavior = behavior_engine.analyze_frame(
                            frame,
                            face_location=face["location"],
                            student_id=student.id,
                        )
                        attendance_engine.log_behavior(student.id, behavior)
                        top, right, _, left = face["location"]
                        anchor_y = top - 10 if top > 36 else top + 24
                        behavior_engine.draw_behavior_overlay(frame, behavior, (left, anchor_y))
                        if behavior.alert:
                            cv2.putText(
                                frame,
                                behavior.alert,
                                (left, min(frame.shape[0] - 18, anchor_y + 26)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    except BehaviorEngineError as exc:
                        cv2.putText(
                            frame,
                            str(exc)[:110],
                            (16, 66),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                frame_bytes = encode_frame(frame)
                if frame_bytes:
                    yield mjpeg(frame_bytes)
        finally:
            camera.release()
            db.session.remove()

    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = User.query.filter(func.lower(User.email) == email).first()
            if user and user.check_password(password) and user.active:
                login_user(user)
                return redirect(request.args.get("next") or url_for("dashboard"))
            flash("Invalid email or password.", "danger")

        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        today_date = date.today()
        start_at, end_at = AttendanceEngine.day_bounds(today_date)
        total_students = Student.query.filter_by(active=True).count()
        present_today = Attendance.query.filter_by(date=today_date).count()
        behavior_summary = attendance_engine.behavior_summary(start_at, end_at)
        recent_attendance = (
            Attendance.query.join(Student)
            .filter(Attendance.date == today_date)
            .order_by(Attendance.time.desc())
            .limit(8)
            .all()
        )
        recent_alerts = (
            BehaviorLog.query.join(Student)
            .filter(BehaviorLog.timestamp >= datetime.now() - timedelta(minutes=15))
            .filter((BehaviorLog.status == "sleepy") | (BehaviorLog.alert.isnot(None)))
            .order_by(BehaviorLog.timestamp.desc())
            .limit(6)
            .all()
        )
        avg_attention = 0
        if behavior_summary:
            avg_attention = round(
                sum(row["average_attention"] for row in behavior_summary) / len(behavior_summary),
                2,
            )
        return render_template(
            "dashboard.html",
            total_students=total_students,
            present_today=present_today,
            avg_attention=avg_attention,
            recent_attendance=recent_attendance,
            recent_alerts=recent_alerts,
            behavior_summary=behavior_summary,
            encodings_loaded=len(face_engine._cache["encodings"]),
        )

    @app.route("/students", methods=["GET", "POST"])
    @login_required
    def students():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            roll_no = request.form.get("roll_no", "").strip()
            email = request.form.get("email", "").strip().lower()

            if not name or not roll_no or not email:
                flash("Name, roll number, and email are required.", "danger")
                return redirect(url_for("students"))

            exists = Student.query.filter(
                (Student.roll_no == roll_no) | (func.lower(Student.email) == email)
            ).first()
            if exists:
                flash("A student with that roll number or email already exists.", "danger")
                return redirect(url_for("students"))

            student = Student(name=name, roll_no=roll_no, email=email)
            db.session.add(student)
            db.session.commit()
            student.dataset_path = str(face_engine.student_dataset_dir(student))
            db.session.commit()
            flash("Student created. Capture face images before training.", "success")
            return redirect(url_for("register_student", student_id=student.id))

        all_students = Student.query.order_by(Student.roll_no.asc()).all()
        return render_template("students.html", students=all_students)

    @app.route("/students/<int:student_id>/register")
    @login_required
    def register_student(student_id: int):
        student = db.session.get(Student, student_id)
        if student is None:
            flash("Student not found.", "danger")
            return redirect(url_for("students"))
        return render_template(
            "register_student.html",
            student=student,
            default_count=max(20, min(50, app.config["CAPTURE_IMAGE_COUNT"])),
        )

    @app.route("/students/<int:student_id>/capture", methods=["POST"])
    @login_required
    def capture_student(student_id: int):
        student = db.session.get(Student, student_id)
        if student is None:
            return jsonify({"error": "Student not found."}), 404

        payload = request.get_json(silent=True) or request.form
        count = capture_count_from_payload(payload)
        camera_index = camera_index_from_payload(payload)
        try:
            result = face_engine.capture_student_images(student, camera_index=camera_index, count=count)
            student.dataset_path = result["dataset_path"]
            db.session.commit()
            return jsonify(result)
        except FaceEngineError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/students/<int:student_id>/train", methods=["POST"])
    @login_required
    def train_student(student_id: int):
        student = db.session.get(Student, student_id)
        if student is None:
            return jsonify({"error": "Student not found."}), 404
        try:
            return jsonify(face_engine.train_student(student))
        except FaceEngineError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/students/<int:student_id>/toggle", methods=["POST"])
    @login_required
    def toggle_student(student_id: int):
        student = db.session.get(Student, student_id)
        if student is None:
            flash("Student not found.", "danger")
            return redirect(url_for("students"))
        student.active = not student.active
        db.session.commit()
        flash("Student status updated.", "success")
        return redirect(url_for("students"))

    @app.route("/train", methods=["POST"])
    @login_required
    def train_all():
        active_students = Student.query.filter_by(active=True).order_by(Student.roll_no.asc()).all()
        try:
            result = face_engine.train_all(active_students)
            flash(f"Training complete. {result['encodings_total']} encodings saved.", "success")
        except FaceEngineError as exc:
            flash(str(exc), "danger")
        return redirect(url_for("students"))

    @app.route("/attendance/live")
    @login_required
    def live_attendance():
        return render_template("live_attendance.html", default_camera=app.config["DEFAULT_CAMERA_INDEX"])

    @app.route("/video/preview")
    @login_required
    def video_preview():
        if cv2 is None:
            return Response("OpenCV is not installed.", status=503)
        camera_index = camera_index_from_request()
        return Response(
            stream_with_context(preview_stream(camera_index)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/video/attendance")
    @login_required
    def video_attendance():
        if cv2 is None:
            return Response("OpenCV is not installed.", status=503)
        camera_index = camera_index_from_request()
        return Response(
            stream_with_context(attendance_stream(camera_index)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/attendance")
    @login_required
    def attendance_records():
        target_date = parse_date(request.args.get("date"), date.today())
        records = attendance_engine.attendance_for_date(target_date)
        return render_template("attendance.html", records=records, selected_date=target_date)

    @app.route("/behavior")
    @login_required
    def behavior():
        target_date = parse_date(request.args.get("date"), date.today())
        start_at, end_at = AttendanceEngine.day_bounds(target_date)
        student_id = request.args.get("student_id", type=int)
        summary = attendance_engine.behavior_summary(start_at, end_at, student_id=student_id)
        logs_query = BehaviorLog.query.join(Student).filter(
            BehaviorLog.timestamp >= start_at,
            BehaviorLog.timestamp < end_at,
        )
        if student_id:
            logs_query = logs_query.filter(BehaviorLog.student_id == student_id)
        logs = logs_query.order_by(BehaviorLog.timestamp.desc()).limit(100).all()
        return render_template(
            "behavior.html",
            selected_date=target_date,
            students=Student.query.order_by(Student.roll_no.asc()).all(),
            selected_student_id=student_id,
            summary=summary,
            logs=logs,
        )

    @app.route("/reports")
    @login_required
    def reports():
        return render_template("reports.html")

    @app.route("/reports/attendance/daily.csv")
    @login_required
    def daily_attendance_csv():
        target_date = parse_date(request.args.get("date"), date.today())
        path = report_engine.daily_attendance_csv(target_date)
        return send_file(path, as_attachment=True, download_name=Path(path).name)

    @app.route("/reports/attendance/monthly.csv")
    @login_required
    def monthly_attendance_csv():
        year, month = parse_month(request.args.get("month"))
        summary = attendance_engine.monthly_summary(year, month)
        path = report_engine.monthly_attendance_csv(year, month, summary)
        return send_file(path, as_attachment=True, download_name=Path(path).name)

    @app.route("/reports/behavior.csv")
    @login_required
    def behavior_csv():
        target_date = parse_date(request.args.get("date"), date.today())
        start_at, end_at = AttendanceEngine.day_bounds(target_date)
        summary = attendance_engine.behavior_summary(start_at, end_at)
        path = report_engine.behavior_csv(start_at, end_at, summary)
        return send_file(path, as_attachment=True, download_name=Path(path).name)

    @app.route("/reports/attendance/daily.pdf")
    @login_required
    def daily_attendance_pdf():
        target_date = parse_date(request.args.get("date"), date.today())
        path = report_engine.daily_attendance_pdf(target_date)
        return send_file(path, as_attachment=True, download_name=Path(path).name)

    @app.route("/reports/behavior.pdf")
    @login_required
    def behavior_pdf():
        target_date = parse_date(request.args.get("date"), date.today())
        start_at, end_at = AttendanceEngine.day_bounds(target_date)
        summary = attendance_engine.behavior_summary(start_at, end_at)
        path = report_engine.behavior_pdf(start_at, end_at, summary)
        return send_file(path, as_attachment=True, download_name=Path(path).name)

    @app.route("/api/login", methods=["POST"])
    def api_login():
        payload = request.get_json(silent=True) or {}
        email = payload.get("email", "").strip().lower()
        password = payload.get("password", "")
        user = User.query.filter(func.lower(User.email) == email).first()
        if not user or not user.check_password(password) or not user.active:
            return jsonify({"error": "Invalid email or password."}), 401
        return jsonify({"token": create_token(user), "user": user.to_dict()})

    @app.route("/api/students", methods=["GET", "POST"])
    @jwt_required
    def api_students():
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            name = payload.get("name", "").strip()
            roll_no = payload.get("roll_no", "").strip()
            email = payload.get("email", "").strip().lower()
            if not name or not roll_no or not email:
                return jsonify({"error": "name, roll_no, and email are required."}), 400
            exists = Student.query.filter(
                (Student.roll_no == roll_no) | (func.lower(Student.email) == email)
            ).first()
            if exists:
                return jsonify({"error": "Student already exists."}), 409
            student = Student(name=name, roll_no=roll_no, email=email)
            db.session.add(student)
            db.session.commit()
            student.dataset_path = str(face_engine.student_dataset_dir(student))
            db.session.commit()
            return jsonify(student.to_dict()), 201

        return jsonify([student.to_dict() for student in Student.query.order_by(Student.roll_no.asc()).all()])

    @app.route("/api/attendance/today")
    @jwt_required
    def api_attendance_today():
        records = attendance_engine.attendance_for_date(date.today())
        return jsonify([record.to_dict() for record in records])

    @app.route("/api/behavior/summary")
    @jwt_required
    def api_behavior_summary():
        target_date = parse_date(request.args.get("date"), date.today())
        start_at, end_at = AttendanceEngine.day_bounds(target_date)
        student_id = request.args.get("student_id", type=int)
        return jsonify(attendance_engine.behavior_summary(start_at, end_at, student_id=student_id))

    @app.route("/api/alerts")
    @jwt_required
    def api_alerts():
        since = datetime.now() - timedelta(minutes=request.args.get("minutes", default=15, type=int))
        alerts = (
            BehaviorLog.query.join(Student)
            .filter(BehaviorLog.timestamp >= since)
            .filter((BehaviorLog.status == "sleepy") | (BehaviorLog.alert.isnot(None)))
            .order_by(BehaviorLog.timestamp.desc())
            .limit(50)
            .all()
        )
        return jsonify([alert.to_dict() for alert in alerts])

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
