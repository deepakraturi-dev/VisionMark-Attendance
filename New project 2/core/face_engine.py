from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Iterable

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import face_recognition
except ImportError:  # pragma: no cover - handled at runtime
    face_recognition = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None


class FaceEngineError(RuntimeError):
    """Raised when face recognition cannot be performed."""


def _require_face_stack() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if face_recognition is None:
        missing.append("face-recognition")
    if np is None:
        missing.append("numpy")
    if missing:
        raise FaceEngineError(
            "Missing required AI dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "student"


def _largest_face(face_locations: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not face_locations:
        return None
    return max(face_locations, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))


class FaceEngine:
    def __init__(
        self,
        dataset_dir: str | Path,
        encodings_path: str | Path,
        tolerance: float = 0.48,
        detection_model: str = "hog",
        frame_scale: float = 0.25,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.encodings_path = Path(encodings_path)
        self.tolerance = tolerance
        self.detection_model = detection_model
        self.frame_scale = frame_scale
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.encodings_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {
            "student_ids": [],
            "names": [],
            "roll_nos": [],
            "encodings": [],
        }
        self.load_encodings()

    def student_dataset_dir(self, student) -> Path:
        return self.dataset_dir / f"{_safe_slug(student.roll_no)}_{student.id}"

    def load_encodings(self) -> dict:
        if not self.encodings_path.exists():
            return self._cache

        with self.encodings_path.open("rb") as file:
            data = pickle.load(file)

        self._cache = {
            "student_ids": data.get("student_ids", []),
            "names": data.get("names", []),
            "roll_nos": data.get("roll_nos", []),
            "encodings": data.get("encodings", []),
        }
        return self._cache

    def save_encodings(self) -> None:
        payload = {
            "version": 1,
            "student_ids": self._cache["student_ids"],
            "names": self._cache["names"],
            "roll_nos": self._cache["roll_nos"],
            "encodings": self._cache["encodings"],
        }
        with self.encodings_path.open("wb") as file:
            pickle.dump(payload, file)

    def has_encodings(self) -> bool:
        return bool(self._cache["encodings"])

    def capture_student_images(
        self,
        student,
        camera_index: int = 0,
        count: int = 30,
        frame_wait_seconds: float = 0.12,
        max_frames: int | None = None,
    ) -> dict:
        _require_face_stack()
        target_dir = self.student_dataset_dir(student)
        target_dir.mkdir(parents=True, exist_ok=True)

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise FaceEngineError(f"Unable to open camera index {camera_index}.")

        captured = 0
        scanned = 0
        max_frames = max_frames or count * 35

        try:
            while captured < count and scanned < max_frames:
                ok, frame = camera.read()
                scanned += 1
                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb, model=self.detection_model)
                if _largest_face(face_locations) is None:
                    continue

                filename = target_dir / f"{student.roll_no}_{int(time.time() * 1000)}_{captured + 1:03d}.jpg"
                cv2.imwrite(str(filename), frame)
                captured += 1
                time.sleep(frame_wait_seconds)
        finally:
            camera.release()

        return {
            "student_id": student.id,
            "dataset_path": str(target_dir),
            "requested": count,
            "captured": captured,
            "frames_scanned": scanned,
        }

    def _encode_image(self, image_path: Path) -> list:
        _require_face_stack()
        image = face_recognition.load_image_file(str(image_path))
        face_locations = face_recognition.face_locations(image, model=self.detection_model)
        face_location = _largest_face(face_locations)
        if face_location is None:
            return []
        return face_recognition.face_encodings(image, known_face_locations=[face_location])

    def _student_image_paths(self, student) -> list[Path]:
        folder = self.student_dataset_dir(student)
        if not folder.exists():
            return []
        image_paths: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(folder.glob(pattern))
        return sorted(image_paths)

    def remove_student_encodings(self, student_id: int) -> None:
        keep = [
            index
            for index, existing_id in enumerate(self._cache["student_ids"])
            if int(existing_id) != int(student_id)
        ]
        self._cache = {
            "student_ids": [self._cache["student_ids"][i] for i in keep],
            "names": [self._cache["names"][i] for i in keep],
            "roll_nos": [self._cache["roll_nos"][i] for i in keep],
            "encodings": [self._cache["encodings"][i] for i in keep],
        }

    def train_student(self, student, replace_existing: bool = True) -> dict:
        _require_face_stack()
        image_paths = self._student_image_paths(student)
        if replace_existing:
            self.remove_student_encodings(student.id)

        encoded_count = 0
        skipped_count = 0
        for image_path in image_paths:
            encodings = self._encode_image(image_path)
            if not encodings:
                skipped_count += 1
                continue
            self._cache["student_ids"].append(student.id)
            self._cache["names"].append(student.name)
            self._cache["roll_nos"].append(student.roll_no)
            self._cache["encodings"].append(encodings[0])
            encoded_count += 1

        self.save_encodings()
        return {
            "student_id": student.id,
            "images_found": len(image_paths),
            "encoded": encoded_count,
            "skipped": skipped_count,
            "encodings_total": len(self._cache["encodings"]),
        }

    def train_all(self, students: Iterable) -> dict:
        _require_face_stack()
        self._cache = {
            "student_ids": [],
            "names": [],
            "roll_nos": [],
            "encodings": [],
        }
        results = []
        for student in students:
            results.append(self.train_student(student, replace_existing=False))
        return {
            "students": results,
            "encodings_total": len(self._cache["encodings"]),
        }

    def recognize_frame(self, frame) -> list[dict]:
        _require_face_stack()
        if frame is None:
            return []

        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        small_locations = face_recognition.face_locations(rgb_small, model=self.detection_model)
        face_encodings = face_recognition.face_encodings(rgb_small, small_locations)

        recognitions = []
        known_encodings = self._cache["encodings"]
        scale = 1 / self.frame_scale

        for face_location, face_encoding in zip(small_locations, face_encodings):
            top, right, bottom, left = [int(value * scale) for value in face_location]
            result = {
                "student_id": None,
                "name": "Unknown",
                "roll_no": None,
                "distance": None,
                "confidence": 0.0,
                "location": (top, right, bottom, left),
                "recognized": False,
            }

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_index = int(np.argmin(distances))
                best_distance = float(distances[best_index])
                if best_distance <= self.tolerance:
                    result.update(
                        {
                            "student_id": int(self._cache["student_ids"][best_index]),
                            "name": self._cache["names"][best_index],
                            "roll_no": self._cache["roll_nos"][best_index],
                            "distance": round(best_distance, 4),
                            "confidence": round(max(0.0, 1.0 - best_distance), 4),
                            "recognized": True,
                        }
                    )

            recognitions.append(result)

        return recognitions

    @staticmethod
    def draw_recognition_overlay(frame, recognitions: list[dict]) -> None:
        if cv2 is None or frame is None:
            return
        for face in recognitions:
            top, right, bottom, left = face["location"]
            recognized = face.get("recognized", False)
            color = (32, 191, 107) if recognized else (46, 134, 222)
            label = face["name"] if recognized else "Unknown"
            if face.get("roll_no"):
                label = f"{label} ({face['roll_no']})"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
