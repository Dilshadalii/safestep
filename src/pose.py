import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Pairs of landmark indices to draw as bones (body only, no face)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]


class PoseEstimator:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.t0 = time.time()
        self.last_ts = -1

    def process(self, frame_bgr):
        """Returns a list of 33 landmarks (normalized x, y, visibility) or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.time() - self.t0) * 1000)
        if ts <= self.last_ts:          # timestamps must strictly increase
            ts = self.last_ts + 1
        self.last_ts = ts

        result = self.landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def close(self):
        self.landmarker.close()


def draw_skeleton(frame, landmarks, min_vis=0.5):
    h, w = frame.shape[:2]
    pts = {}
    for i, lm in enumerate(landmarks):
        if getattr(lm, "visibility", 1.0) is None or lm.visibility >= min_vis:
            pts[i] = (int(lm.x * w), int(lm.y * h))

    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for i, p in pts.items():
        if i >= 11:  # skip face points
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return frame