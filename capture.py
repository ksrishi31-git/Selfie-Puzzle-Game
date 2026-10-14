"""
capture.py
----------
Manages frame selection using hand gestures and captures the selected image region.

Interaction flow:
1. No selection: first pinch sets the ANCHOR point (top-left corner).
2. Drag mode: while pinching, the current pinch position defines the second corner,
   dynamically drawing the selection rectangle.
3. Release pinch: rectangle is locked in place.
4. Second pinch INSIDE the locked rectangle: starts a 2-second countdown.
5. After 2 seconds: captures the image.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple

STATE_IDLE      = "idle"
STATE_DRAWING   = "drawing"
STATE_LOCKED    = "locked"
STATE_COUNTDOWN = "countdown"
STATE_CAPTURED  = "captured"

COUNTDOWN_SECONDS = 3


class FrameCapture:
    """
    Handles the interactive rectangular region selection and image capture.
    """

    def __init__(self):
        self.state = STATE_IDLE
        self.anchor: Optional[Tuple[int, int]] = None
        self.current_pos: Optional[Tuple[int, int]] = None
        self.captured_image: Optional[np.ndarray] = None
        self.rect: Optional[Tuple[int, int, int, int]] = None

        self.min_rect_size  = 80
        self._flash_frames  = 0
        self._flash_duration = 8

        self._countdown_start: Optional[float] = None  # time.time() when countdown began

    def update(
        self,
        frame: np.ndarray,
        is_pinching: bool,
        pinch_start: bool,
        pinch_center: Optional[Tuple[int, int]],
    ) -> bool:
        just_captured = False

        if self.state == STATE_IDLE:
            if is_pinching and pinch_center:
                self.anchor = pinch_center
                self.current_pos = pinch_center
                self.state = STATE_DRAWING

        elif self.state == STATE_DRAWING:
            if is_pinching and pinch_center:
                self.current_pos = pinch_center
            else:
                if self.anchor and self.current_pos:
                    x1 = min(self.anchor[0], self.current_pos[0])
                    y1 = min(self.anchor[1], self.current_pos[1])
                    x2 = max(self.anchor[0], self.current_pos[0])
                    y2 = max(self.anchor[1], self.current_pos[1])

                    if (x2 - x1) >= self.min_rect_size and (y2 - y1) >= self.min_rect_size:
                        self.rect = (x1, y1, x2, y2)
                        self.state = STATE_LOCKED
                    else:
                        self._reset_selection()

        elif self.state == STATE_LOCKED:
            if pinch_start and pinch_center and self.rect:
                x1, y1, x2, y2 = self.rect
                px, py = pinch_center
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # Start the 2-second countdown
                    self._countdown_start = time.time()
                    self.state = STATE_COUNTDOWN
                else:
                    self._reset_selection()

        elif self.state == STATE_COUNTDOWN:
            elapsed = time.time() - self._countdown_start
            if elapsed >= COUNTDOWN_SECONDS:
                # Time's up — capture now
                if self.rect:
                    x1, y1, x2, y2 = self.rect
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        self.captured_image = cropped.copy()
                        self.state = STATE_CAPTURED
                        self._flash_frames = self._flash_duration
                        just_captured = True

        return just_captured

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        self._draw_instructions(frame, h, w)

        if self.state == STATE_DRAWING and self.anchor and self.current_pos:
            x1 = min(self.anchor[0], self.current_pos[0])
            y1 = min(self.anchor[1], self.current_pos[1])
            x2 = max(self.anchor[0], self.current_pos[0])
            y2 = max(self.anchor[1], self.current_pos[1])

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 150, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 150, 255), 3)
            self._draw_corner_handles(frame, x1, y1, x2, y2, (50, 150, 255))

        elif self.state == STATE_LOCKED and self.rect:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 100), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 3)
            self._draw_corner_handles(frame, x1, y1, x2, y2, (0, 220, 100))

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self._put_centered_text(frame, "Pinch inside to start capture!", cx, cy, (255, 255, 255), 0.7)

        elif self.state == STATE_COUNTDOWN and self.rect:
            x1, y1, x2, y2 = self.rect
            elapsed  = time.time() - self._countdown_start
            remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)

            # Pulse the border red as countdown runs
            pulse = int(127 + 127 * abs(np.sin(elapsed * math.pi)))
            border_color = (0, 255 - pulse, pulse)   # shifts green -> red

            cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 4)
            self._draw_corner_handles(frame, x1, y1, x2, y2, border_color)

            # Big countdown number in centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self._draw_countdown_number(frame, remaining, cx, cy)

        elif self.state == STATE_CAPTURED:
            if self._flash_frames > 0:
                alpha = self._flash_frames / self._flash_duration
                white = np.ones_like(frame) * 255
                cv2.addWeighted(white, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)
                self._flash_frames -= 1

        return frame

    def _draw_countdown_number(self, frame, remaining: float, cx: int, cy: int):
        """Draw a large countdown number with a circular progress ring."""
        import math

        # Circle ring
        radius   = 55
        thick    = 6
        progress = 1.0 - (remaining / COUNTDOWN_SECONDS)   # 0 -> 1 as time passes
        angle    = int(360 * progress)

        # Background ring (dark)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360, (40, 40, 40), thick)
        # Foreground arc
        arc_color = (80, 220, 120) if remaining > 0.5 else (60, 120, 255)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, arc_color, thick)

        # Number
        num_text = str(int(remaining) + 1)   # shows 2 then 1
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 2.0
        (tw, th), _ = cv2.getTextSize(num_text, font, scale, 3)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(frame, num_text, (tx + 2, ty + 2), font, scale, (0, 0, 0), 5)
        cv2.putText(frame, num_text, (tx, ty), font, scale, (255, 255, 255), 3)

        # "Get ready!" below the ring
        self._put_centered_text(frame, "Get ready!", cx, cy + radius + 22, (200, 200, 200), 0.65)

    def _draw_instructions(self, frame: np.ndarray, h: int, w: int):
        instructions = {
            STATE_IDLE:      "Pinch & drag to draw selection box",
            STATE_DRAWING:   "Release pinch to lock the box",
            STATE_LOCKED:    "Pinch inside box to capture | Pinch outside to reset",
            STATE_COUNTDOWN: "Hold still! Capturing in...",
            STATE_CAPTURED:  "Image captured!",
        }
        text = instructions.get(self.state, "")
        colors = {
            STATE_IDLE:      (200, 200, 200),
            STATE_DRAWING:   (50, 150, 255),
            STATE_LOCKED:    (0, 220, 100),
            STATE_COUNTDOWN: (60, 180, 255),
            STATE_CAPTURED:  (50, 255, 150),
        }
        color = colors.get(self.state, (200, 200, 200))

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = (w - tw) // 2
        ty = 30
        cv2.rectangle(frame, (tx - 10, ty - th - 6), (tx + tw + 10, ty + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_corner_handles(self, frame, x1, y1, x2, y2, color):
        length    = 20
        thickness = 4
        corners   = [
            ((x1, y1), ( 1,  1)),
            ((x2, y1), (-1,  1)),
            ((x1, y2), ( 1, -1)),
            ((x2, y2), (-1, -1)),
        ]
        for (cx, cy), (dx, dy) in corners:
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness)

    def _put_centered_text(self, frame, text, cx, cy, color, scale=0.7):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(frame, text, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    def _reset_selection(self):
        self.state             = STATE_IDLE
        self.anchor            = None
        self.current_pos       = None
        self.rect              = None
        self._countdown_start  = None

    def reset(self):
        self._reset_selection()
        self.captured_image = None

    def is_done(self) -> bool:
        return self.state == STATE_CAPTURED and self.captured_image is not None

    def get_captured_image(self) -> Optional[np.ndarray]:
        return self.captured_image


# math is used in draw_overlay for the pulse calculation
import math