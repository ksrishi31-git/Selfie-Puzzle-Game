"""
camera.py
---------
Handles webcam initialization, frame capture, and release.
Provides a clean interface for accessing the user's webcam with error handling.
"""

import cv2
import numpy as np


class Camera:
    """
    Manages webcam access using OpenCV.

    Attributes:
        camera_index (int): Index of the webcam device (default: 0).
        width (int): Desired frame width.
        height (int): Desired frame height.
        fps (int): Desired frames per second.
        cap (cv2.VideoCapture): OpenCV video capture object.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open webcam at index {self.camera_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Camera opened: {self.width}x{self.height} @ {self.fps} FPS")
        return True

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Flip horizontally so the user sees a mirror view
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[INFO] Camera released.")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_resolution(self):
        return self.width, self.height
