import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import urllib.request
import os
from typing import Optional, Tuple

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model():
    """Download the hand landmarker model if it doesn't exist locally."""
    if not os.path.exists(_MODEL_PATH):
        print("[INFO] Downloading hand_landmarker.task model (~20 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model downloaded.")


# Simple result container that mirrors the old mp.solutions.hands result shape
class _HandResults:
    def __init__(self, detection_result):
        self._result = detection_result

    @property
    def multi_hand_landmarks(self):
        if not self._result or not self._result.hand_landmarks:
            return None
        return self._result.hand_landmarks  # list of lists of NormalizedLandmark


class GestureDetector:
    """
    Detects hand landmarks and pinch gestures using MediaPipe Hands (Tasks API).
    """

    THUMB_TIP = 4
    INDEX_TIP = 8

    # Colours for drawing connections
    _CONNECTION_COLOR = (0, 200, 100)
    _LANDMARK_COLOR = (255, 255, 255)

    # All 21-landmark connections (same as mp.solutions.hands.HAND_CONNECTIONS)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),              # Palm
    ]

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        pinch_threshold: float = 0.06,
        pinch_cooldown: int = 20,
    ):
        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0  # monotonic timestamp in ms

        self.pinch_threshold = pinch_threshold
        self.pinch_cooldown = pinch_cooldown
        self._cooldown_counter = 0
        self._was_pinching = False

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> _HandResults:
        """Run hand landmark detection on a BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Tasks API VIDEO mode requires strictly increasing timestamps
        self._frame_ts_ms += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        return _HandResults(result)

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------

    def get_landmark_positions(
        self, results: _HandResults, frame_width: int, frame_height: int
    ) -> Optional[dict]:
        if not results.multi_hand_landmarks:
            return None
        landmarks = results.multi_hand_landmarks[0]
        return {
            idx: (int(lm.x * frame_width), int(lm.y * frame_height))
            for idx, lm in enumerate(landmarks)
        }

    def get_pinch_center(
        self, results: _HandResults, frame_width: int, frame_height: int
    ) -> Optional[Tuple[int, int]]:
        positions = self.get_landmark_positions(results, frame_width, frame_height)
        if positions is None:
            return None
        thumb = positions[self.THUMB_TIP]
        index = positions[self.INDEX_TIP]
        return (thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2

    # ------------------------------------------------------------------
    # Pinch detection
    # ------------------------------------------------------------------

    def is_pinching(
        self, results: _HandResults, frame_width: int, frame_height: int
    ) -> bool:
        if not results.multi_hand_landmarks:
            return False
        lm = results.multi_hand_landmarks[0]
        thumb = lm[self.THUMB_TIP]
        index = lm[self.INDEX_TIP]
        dist = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)
        return dist < self.pinch_threshold

    def detected_pinch_start(
        self, results: _HandResults, frame_width: int, frame_height: int
    ) -> bool:
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        currently_pinching = self.is_pinching(results, frame_width, frame_height)
        pinch_started = currently_pinching and not self._was_pinching
        self._was_pinching = currently_pinching

        if pinch_started and self._cooldown_counter == 0:
            self._cooldown_counter = self.pinch_cooldown
            return True
        return False

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame: np.ndarray, results: _HandResults) -> np.ndarray:
        if not results.multi_hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for hand in results.multi_hand_landmarks:
            # Draw connections
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                pt1 = (int(hand[start_idx].x * w), int(hand[start_idx].y * h))
                pt2 = (int(hand[end_idx].x * w), int(hand[end_idx].y * h))
                cv2.line(frame, pt1, pt2, self._CONNECTION_COLOR, 2)
            # Draw landmark dots
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, self._LANDMARK_COLOR, -1)
                cv2.circle(frame, (cx, cy), 4, self._CONNECTION_COLOR, 1)
        return frame

    def draw_pinch_indicator(
        self, frame: np.ndarray, results: _HandResults, frame_width: int, frame_height: int
    ) -> np.ndarray:
        if self.is_pinching(results, frame_width, frame_height):
            center = self.get_pinch_center(results, frame_width, frame_height)
            if center:
                cv2.circle(frame, center, 15, (0, 255, 0), -1)
                cv2.circle(frame, center, 18, (255, 255, 255), 2)
        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._landmarker.close()