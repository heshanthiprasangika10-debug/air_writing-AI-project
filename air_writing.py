"""
Air Writing AI  -  real-time air writing with Computer Vision
Python + OpenCV + MediaPipe + NumPy

Gestures (one hand, mirror view):
  Index finger only up ............ DRAW
  Index + Middle fingers up ....... SELECT (move pointer, pick colour / eraser / clear from top toolbar)
  Open palm (all 5 fingers) hold .. CLEAR the canvas (hold ~1 sec)
  Any other pose .................. pen up (nothing is drawn)

Keys:
  c  clear      u  undo      s  save PNG      r  recognise text (needs pytesseract)
  +/-  pen size      h  show/hide hand skeleton      q / ESC  quit

Install:
  pip install opencv-python numpy mediapipe==0.10.14
  (optional text recognition)  pip install pytesseract   + install Tesseract-OCR program
"""
import argparse
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TOOLBAR_H = 80                 # height of top toolbar (pixels)
SMOOTHING = 0.45               # 0..1  (higher = follows finger faster, lower = smoother)
MAX_JUMP = 250                 # ignore line if pointer jumps more than this (tracking glitch)
CLEAR_HOLD_SEC = 1.0           # hold open palm this long to clear
SELECT_COOLDOWN = 0.7          # seconds between toolbar clicks
UNDO_LIMIT = 30

# name -> BGR colour.  "Eraser" and "Clear" / "Undo" are special buttons
BUTTONS = [
    ("Blue", (255, 80, 0)),
    ("Green", (0, 200, 0)),
    ("Red", (0, 0, 255)),
    ("Yellow", (0, 230, 255)),
    ("White", (255, 255, 255)),
    ("Eraser", (60, 60, 60)),
    ("Undo", (120, 90, 40)),
    ("Clear", (40, 40, 160)),
]


# ----------------------------------------------------------------------------
# Hand tracking (MediaPipe)
# ----------------------------------------------------------------------------
class HandTracker:
    """Wraps MediaPipe Hands. detect(frame_bgr) -> list of 21 (x, y) pixel points or None."""

    def __init__(self):
        import mediapipe as mp  # imported here so the rest of the file can be tested without it

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.drawer = mp.solutions.drawing_utils
        self.connections = mp.solutions.hands.HAND_CONNECTIONS
        self._last = None

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        self._last = result
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark
        return [(int(p.x * w), int(p.y * h)) for p in lm]

    def draw_skeleton(self, frame):
        if self._last and self._last.multi_hand_landmarks:
            for hand in self._last.multi_hand_landmarks:
                self.drawer.draw_landmarks(frame, hand, self.connections)

    def close(self):
        self.hands.close()


# ----------------------------------------------------------------------------
# Gesture logic (pure NumPy / math - no MediaPipe needed)
# ----------------------------------------------------------------------------
def fingers_up(pts):
    """Return [thumb, index, middle, ring, pinky] as 1/0 from 21 landmark points.

    Landmarks: 0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
    A finger is 'up' when its tip is higher on screen (smaller y) than its PIP joint.
    Thumb moves sideways, so compare tip distance vs. IP distance from the pinky base (17).
    """
    p = np.array(pts, dtype=float)
    state = []
    thumb_up = np.linalg.norm(p[4] - p[17]) > np.linalg.norm(p[3] - p[17]) * 1.05
    state.append(int(thumb_up))
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        state.append(int(p[tip][1] < p[pip][1]))
    return state


def classify(state):
    """Map finger state to a mode name."""
    thumb, index, middle, ring, pinky = state
    if index and not middle and not ring and not pinky:
        return "DRAW"
    if index and middle and not ring and not pinky:
        return "SELECT"
    if index and middle and ring and pinky and thumb:
        return "PALM"
    return "IDLE"


# ----------------------------------------------------------------------------
# Canvas + UI helpers
# ----------------------------------------------------------------------------
def button_rects(width):
    bw = width // len(BUTTONS)
    return [(i * bw, (i + 1) * bw) for i in range(len(BUTTONS))]


def draw_toolbar(frame, active_name, hover_name):
    rects = button_rects(frame.shape[1])
    for (x0, x1), (name, col) in zip(rects, BUTTONS):
        cv2.rectangle(frame, (x0, 0), (x1, TOOLBAR_H), col, -1)
        if name == active_name:
            cv2.rectangle(frame, (x0 + 3, 3), (x1 - 3, TOOLBAR_H - 3), (255, 255, 255), 4)
        if name == hover_name:
            cv2.rectangle(frame, (x0, 0), (x1, TOOLBAR_H), (0, 255, 255), 2)
        txt_col = (0, 0, 0) if name in ("Yellow", "White", "Green") else (255, 255, 255)
        size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, name, (x0 + (x1 - x0 - size[0]) // 2, TOOLBAR_H // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2, cv2.LINE_AA)


def button_at(x, width):
    for (x0, x1), (name, _) in zip(button_rects(width), BUTTONS):
        if x0 <= x < x1:
            return name
    return None


def merge_canvas(frame, canvas):
    """Put the drawing on top of the camera frame."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    out = frame.copy()
    out[mask] = canvas[mask]
    return out


def canvas_to_white_bg(canvas):
    """Black background + coloured strokes  ->  white background + black strokes."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return 255 - bw


def recognise_text(canvas):
    """Optional OCR. Returns recognised text or a help message."""
    try:
        import pytesseract
    except ImportError:
        return "pytesseract not installed"
    img = canvas_to_white_bg(canvas)
    ys, xs = np.where(img < 128)
    if len(xs) == 0:
        return "(nothing written)"
    pad = 30
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    crop = img[y0:y1, x0:x1]
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    try:
        text = pytesseract.image_to_string(crop, config="--psm 7").strip()
    except Exception as exc:  # Tesseract program missing etc.
        return f"OCR error: {exc.__class__.__name__}"
    return text or "(could not read)"


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def run(source=0, tracker=None, show=True, max_frames=None):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Camera / video could not be opened")
    tracker = tracker or HandTracker()

    canvas = None
    history = []
    colour_name, colour = "Blue", BUTTONS[0][1]
    pen_size, eraser_size = 6, 45
    prev = None
    smooth = None
    last_select = 0.0
    palm_start = None
    show_skeleton = False
    recognised = ""
    t_prev = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)               # mirror, so it feels natural
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)

        pts = tracker.detect(frame)
        mode = "NO HAND"
        hover = None

        if pts:
            state = fingers_up(pts)
            mode = classify(state)
            tip = pts[8]                          # index fingertip
            smooth = tip if smooth is None else (
                int(SMOOTHING * tip[0] + (1 - SMOOTHING) * smooth[0]),
                int(SMOOTHING * tip[1] + (1 - SMOOTHING) * smooth[1]),
            )

            if mode == "DRAW":
                palm_start = None
                if smooth[1] > TOOLBAR_H:        # never draw on the toolbar
                    if prev is None:             # a new stroke begins -> save undo snapshot
                        history.append(canvas.copy())
                        history[:] = history[-UNDO_LIMIT:]
                    elif math.dist(prev, smooth) < MAX_JUMP:
                        is_eraser = colour_name == "Eraser"
                        cv2.line(canvas, prev, smooth,
                                 (0, 0, 0) if is_eraser else colour,
                                 eraser_size if is_eraser else pen_size, cv2.LINE_AA)
                    prev = smooth
                else:
                    prev = None
            else:
                prev = None

            if mode == "SELECT":
                palm_start = None
                if smooth[1] < TOOLBAR_H:
                    hover = button_at(smooth[0], w)
                    if hover and time.time() - last_select > SELECT_COOLDOWN:
                        last_select = time.time()
                        if hover == "Clear":
                            history.append(canvas.copy())
                            canvas[:] = 0
                        elif hover == "Undo":
                            if history:
                                canvas = history.pop()
                        else:
                            colour_name = hover
                            colour = dict(BUTTONS)[hover]

            if mode == "PALM":
                palm_start = palm_start or time.time()
                if time.time() - palm_start >= CLEAR_HOLD_SEC:
                    history.append(canvas.copy())
                    canvas[:] = 0
                    palm_start = None
            elif mode not in ("SELECT", "DRAW"):
                palm_start = None
        else:
            prev = None
            smooth = None
            palm_start = None

        # ---------------- render ----------------
        out = merge_canvas(frame, canvas)
        if show_skeleton and hasattr(tracker, "draw_skeleton"):
            tracker.draw_skeleton(out)
        draw_toolbar(out, colour_name, hover)
        if smooth is not None:
            cv2.circle(out, smooth, 10 if mode == "DRAW" else 14,
                       colour if colour_name != "Eraser" else (200, 200, 200),
                       -1 if mode == "DRAW" else 2, cv2.LINE_AA)
        if palm_start:
            frac = min((time.time() - palm_start) / CLEAR_HOLD_SEC, 1.0)
            cv2.rectangle(out, (20, h - 40), (20 + int(300 * frac), h - 25), (0, 0, 255), -1)
            cv2.putText(out, "Clearing...", (330, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(out, f"Mode: {mode}   FPS: {fps:4.1f}   Pen: {pen_size}", (10, TOOLBAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if recognised:
            cv2.putText(out, f"Text: {recognised}", (10, TOOLBAR_H + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        frames += 1
        if show:
            cv2.imshow("Air Writing AI", out)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255
        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            history.append(canvas.copy())
            canvas[:] = 0
        elif key == ord("u") and history:
            canvas = history.pop()
        elif key == ord("s"):
            name = datetime.now().strftime("airwriting_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, canvas_to_white_bg(canvas))
            print("Saved", os.path.abspath(name))
        elif key == ord("r"):
            recognised = recognise_text(canvas)
            print("Recognised:", recognised)
        elif key in (ord("+"), ord("=")):
            pen_size = min(pen_size + 2, 30)
        elif key in (ord("-"), ord("_")):
            pen_size = max(pen_size - 2, 2)
        elif key == ord("h"):
            show_skeleton = not show_skeleton
        if max_frames and frames >= max_frames:
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if hasattr(tracker, "close"):
        tracker.close()
    return canvas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Air Writing AI")
    ap.add_argument("--source", default="0", help="camera index (0,1..) or a video file path")
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    run(src)
