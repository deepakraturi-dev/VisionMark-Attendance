# AI-Based Smart Attendance System with Behavior Analysis

Production-ready Flask application for student registration, webcam-based face recognition attendance, real-time behavior analysis, dashboards, REST APIs, and CSV/PDF reports.

## Step 1: requirements.txt

Install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`face-recognition` and `dlib` may require CMake and Visual Studio Build Tools on Windows.

## Step 2: folder structure

```text
app.py
config.py
core/
  attendance_engine.py
  behavior_engine.py
  face_engine.py
  report_engine.py
database/
  db.py
  models.py
templates/
static/
dataset/
encodings/
reports/
```

## Step 3: database models

Implemented in `database/models.py`:

- `User`: admin/teacher login with hashed passwords.
- `Student`: name, roll number, email, dataset folder, active status.
- `Attendance`: one row per student per day with date/time and source camera.
- `BehaviorLog`: timestamped attention score, status, EAR, head pose, and alert.

SQLite is used by default at `database/smart_attendance.db`. For MySQL, set `DATABASE_URL` in `.env`, for example `mysql+pymysql://user:password@localhost/smart_attendance`.

## Step 4: face recognition module

Implemented in `core/face_engine.py`:

- Captures 20-50 face images from webcam.
- Stores datasets under `dataset/<roll_no>_<student_id>/`.
- Generates face encodings and saves them to `encodings/faces.pkl`.
- Recognizes faces in real-time with configurable tolerance.

## Step 5: behavior analysis module

Implemented in `core/behavior_engine.py`:

- Uses facial landmarks to compute Eye Aspect Ratio.
- Tracks consecutive low-EAR frames for drowsiness.
- Estimates head pose with `cv2.solvePnP`.
- Returns attention score from 0-100 and status: `attentive`, `distracted`, `sleepy`, or `unknown`.

## Step 6: attendance module

Implemented in `core/attendance_engine.py`:

- Marks attendance once per student per day.
- Persists to SQLite and appends daily CSV files in `reports/`.
- Logs behavior every configured interval.
- Provides daily and monthly summaries.

## Step 7: Flask routes and REST APIs

Implemented in `app.py`:

- `/login`, `/logout`, `/dashboard`
- `/students`, `/students/<id>/register`
- `/students/<id>/capture`, `/students/<id>/train`
- `/attendance/live`, `/video/attendance`
- `/attendance`, `/behavior`, `/reports`
- `/api/login`, `/api/students`, `/api/attendance/today`, `/api/behavior/summary`, `/api/alerts`

JWT API usage:

```powershell
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/login -ContentType application/json -Body '{"email":"admin@example.com","password":"admin123"}').token
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/attendance/today -Headers @{Authorization="Bearer $token"}
```

## Step 8: frontend templates

Implemented in `templates/` and `static/`:

- Login page
- Dashboard with metrics, recent attendance, alerts, and Chart.js behavior overview
- Student registration, webcam preview, dataset capture, and training actions
- Live attendance camera feed
- Attendance records
- Behavior analytics with charts
- CSV/PDF report downloads

## Step 9: how to run

Copy `.env.example` to `.env` and edit secrets/passwords:

```powershell
Copy-Item .env.example .env
```

Start the app:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Default login:

```text
Email: admin@example.com
Password: admin123
```

Workflow:

1. Add a student from `Students`.
2. Open the student dataset page.
3. Capture 20-50 images with the webcam.
4. Train that student or use `Train All`.
5. Open `Live Attendance` and start the camera feed.
6. View attendance, behavior analytics, and reports from the sidebar.
