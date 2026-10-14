import sys
import threading
import cv2
from camera import Camera
from gesture import GestureDetector
from capture import FrameCapture
from puzzle import Puzzle
from game import GameUI
WEBCAM_WINDOW_NAME = "Selfie Puzzle Game - Select Region"

def _build_puzzle_async(image, puzzle: Puzzle, ready_event: threading.Event):
    """Runs in a background thread: slice + shuffle the puzzle, then signal ready."""
    puzzle.create_from_image(image, piece_size=160)
    puzzle.shuffle()
    ready_event.set()

def run_capture_phase():
    camera = Camera(camera_index=0, width=1280, height=720, fps=30)
    if not camera.open():
        print("[ERROR] Failed to open webcam. Check camera connection and permissions.")
        return None, None

    width, height = camera.get_resolution()

    gesture = GestureDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        pinch_threshold=0.06,
        pinch_cooldown=15,
    )

    frame_capture = FrameCapture()

    cv2.namedWindow(WEBCAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WEBCAM_WINDOW_NAME, width, height)

    print("[INFO] Webcam phase started.")
    print("[INFO] Instructions:")
    print("  1. Pinch and drag to draw a selection rectangle.")
    print("  2. Release the pinch to lock the rectangle.")
    print("  3. Pinch inside the rectangle to capture the image.")
    print("  4. Press 'Q' or 'ESC' to quit.")

    captured_image = None
    puzzle = Puzzle()
    puzzle_ready = threading.Event()
    build_thread = None

    while True:
        success, frame = camera.read_frame()
        if not success or frame is None:
            print("[WARNING] Failed to read frame. Retrying...")
            continue

        results = gesture.process(frame)

        is_pinching = gesture.is_pinching(results, width, height)
        pinch_start = gesture.detected_pinch_start(results, width, height)
        pinch_center = gesture.get_pinch_center(results, width, height)

        just_captured = frame_capture.update(
            frame, is_pinching, pinch_start, pinch_center
        )

        # ── Start building the puzzle immediately in the background ──
        if just_captured and build_thread is None:
            captured_image = frame_capture.get_captured_image()
            print("[INFO] Image captured! Building puzzle in background...")
            build_thread = threading.Thread(
                target=_build_puzzle_async,
                args=(captured_image, puzzle, puzzle_ready),
                daemon=True,
            )
            build_thread.start()

        gesture.draw_landmarks(frame, results)
        gesture.draw_pinch_indicator(frame, results, width, height)
        frame_capture.draw_overlay(frame)

        cv2.imshow(WEBCAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[INFO] Quit requested by user.")
            captured_image = None
            break

        # Exit the webcam loop once capture is confirmed
        if frame_capture.is_done():
            break

    cv2.destroyAllWindows()
    gesture.close()
    camera.release()

    if captured_image is None:
        return None, None

    # Wait for puzzle to finish building (usually already done by now)
    if not puzzle_ready.is_set():
        print("[INFO] Waiting for puzzle to finish building...")
        puzzle_ready.wait()

    return captured_image, puzzle


def run_puzzle_phase(puzzle: Puzzle):
    print(f"[INFO] Puzzle ready: {puzzle.piece_size[0]}x{puzzle.piece_size[1]} px per piece.")
    print("[INFO] Launching Pygame game window...")
    game = GameUI(puzzle)
    game.run()
    print("[INFO] Game closed.")


def main():
    print("=" * 55)
    print("   Selfie Puzzle Game")
    print("=" * 55)

    captured_image, puzzle = run_capture_phase()

    if captured_image is None or puzzle is None:
        print("[INFO] No image captured. Exiting.")
        sys.exit(0)

    run_puzzle_phase(puzzle)
    print("[INFO] Thanks for playing!")


if __name__ == "__main__":
    main()