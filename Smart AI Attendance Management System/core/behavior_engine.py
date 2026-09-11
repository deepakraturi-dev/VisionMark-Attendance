from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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


class BehaviorEngineError(RuntimeError):
    """Raised when behavior analysis cannot be performed."""


@dataclass
class BehaviorResult:
    student_id: int | None
    attention_score: int
    status: str
    ear: float | None
    head_yaw: float | None
    head_pitch: float | None
    head_roll: float | None
    looking_away: bool
    sleepy: bool
    alert: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_behavior_stack() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if face_recognition is None:
        missing.append("face-recognition")
    if np is None:
        missing.append("numpy")
    if missing:
        raise BehaviorEngineError(
            "Missing required behavior-analysis dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )


class BehaviorEngine:
    def __init__(
        self,
        ear_threshold: float = 0.22,
        drowsy_frames: int = 12,
        yaw_threshold: float = 25.0,
        pitch_threshold: float = 22.0,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self._low_ear_counts: dict[str, int] = {}

    @staticmethod
    def eye_aspect_ratio(eye_points: list[tuple[int, int]]) -> float | None:
        if np is None or len(eye_points) < 6:
            return None
        points = np.array(eye_points, dtype="float")
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        if horizontal == 0:
            return None
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _average_ear(self, landmarks: dict) -> float | None:
        left_ear = self.eye_aspect_ratio(landmarks.get("left_eye", []))
        right_ear = self.eye_aspect_ratio(landmarks.get("right_eye", []))
        ears = [value for value in (left_ear, right_ear) if value is not None]
        if not ears:
            return None
        return float(sum(ears) / len(ears))

    @staticmethod
    def _point_average(points: list[tuple[int, int]]) -> tuple[float, float] | None:
        if not points or np is None:
            return None
        values = np.array(points, dtype="double")
        mean = values.mean(axis=0)
        return float(mean[0]), float(mean[1])

    def estimate_head_pose(self, landmarks: dict, frame_shape: tuple[int, int, int]) -> dict[str, float] | None:
        if cv2 is None or np is None:
            return None

        required = ("nose_tip", "chin", "left_eye", "right_eye", "top_lip")
        if any(not landmarks.get(name) for name in required):
            return None

        nose_tip = self._point_average(landmarks["nose_tip"])
        chin = landmarks["chin"][8] if len(landmarks["chin"]) > 8 else None
        left_eye = landmarks["left_eye"][0]
        right_eye = landmarks["right_eye"][3] if len(landmarks["right_eye"]) > 3 else landmarks["right_eye"][-1]
        mouth_left = landmarks["top_lip"][0]
        mouth_right = landmarks["top_lip"][6] if len(landmarks["top_lip"]) > 6 else landmarks["top_lip"][-1]
        if nose_tip is None or chin is None:
            return None

        image_points = np.array(
            [nose_tip, chin, left_eye, right_eye, mouth_left, mouth_right],
            dtype="double",
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype="double",
        )

        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = [float(value) for value in euler_angles.flatten()]

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
        }

    def analyze_frame(
        self,
        frame,
        face_location: tuple[int, int, int, int] | None = None,
        student_id: int | None = None,
    ) -> BehaviorResult:
        _require_behavior_stack()
        if frame is None:
            return BehaviorResult(student_id, 0, "unknown", None, None, None, None, False, False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        known_locations = [face_location] if face_location else None
        landmarks_list = face_recognition.face_landmarks(rgb, face_locations=known_locations)

        if not landmarks_list:
            return BehaviorResult(
                student_id=student_id,
                attention_score=35,
                status="unknown",
                ear=None,
                head_yaw=None,
                head_pitch=None,
                head_roll=None,
                looking_away=False,
                sleepy=False,
                alert="Face landmarks not detected",
            )

        landmarks = landmarks_list[0]
        ear = self._average_ear(landmarks)
        pose = self.estimate_head_pose(landmarks, frame.shape) or {}

        key = str(student_id or "anonymous")
        if ear is not None and ear < self.ear_threshold:
            self._low_ear_counts[key] = self._low_ear_counts.get(key, 0) + 1
        else:
            self._low_ear_counts[key] = 0

        sleepy = self._low_ear_counts.get(key, 0) >= self.drowsy_frames
        yaw = pose.get("yaw")
        pitch = pose.get("pitch")
        roll = pose.get("roll")
        looking_away = bool(
            (yaw is not None and abs(yaw) > self.yaw_threshold)
            or (pitch is not None and abs(pitch) > self.pitch_threshold)
        )

        score = 100
        if ear is None:
            score -= 20
        elif sleepy:
            score -= 45
        elif ear < self.ear_threshold:
            score -= 25

        if looking_away:
            score -= 30
        if yaw is None or pitch is None:
            score -= 10

        score = max(0, min(100, int(score)))

        if sleepy:
            status = "sleepy"
            alert = "Student appears drowsy"
        elif looking_away:
            status = "distracted"
            alert = "Student is looking away"
        else:
            status = "attentive"
            alert = None

        return BehaviorResult(
            student_id=student_id,
            attention_score=score,
            status=status,
            ear=round(ear, 4) if ear is not None else None,
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
            looking_away=looking_away,
            sleepy=sleepy,
            alert=alert,
        )

    @staticmethod
    def draw_behavior_overlay(frame, result: BehaviorResult, anchor: tuple[int, int] = (12, 32)) -> None:
        if cv2 is None or frame is None:
            return

        color = (32, 191, 107)
        if result.status == "distracted":
            color = (0, 165, 255)
        elif result.status == "sleepy":
            color = (0, 0, 255)

        text = f"{result.status.title()} | Score {result.attention_score}"
        cv2.putText(
            frame,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
