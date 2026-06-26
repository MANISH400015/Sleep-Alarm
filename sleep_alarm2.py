"""
Sleep Alarm v3 — Eye Monitoring Study Assistant (Enhanced)
============================================================
Uses webcam + MediaPipe Face Mesh to detect drowsiness and alert the user.

New in v3:
  • PERCLOS metric (clinical drowsiness standard)
  • Yawn counter + yawn-rate alarm
  • Micro-sleep detection & logging
  • Drowsiness Score bar (0-100, weighted composite)
  • EAR sparkline graph
  • Session timeline strip (focus / distracted / alarm)
  • Pomodoro break reminders
  • Mute toggle (S key), live EAR threshold (+/-)
  • Hold SPACE 1 s to dismiss alarm (anti-accidental)
  • Animated pulse ring on alarm
  • Clock, next-break countdown in HUD
  • Richer JSON session report

Dependencies:
    pip install opencv-python mediapipe numpy pygame

Usage:
    python sleep_alarm2.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
import pygame
import time
import os
import sys
import json
import datetime
import math
import urllib.request
from dataclasses import dataclass, field
from collections import deque

# Force UTF-8 output so box-drawing / arrow chars work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════
#  CONFIG  (edit these to tune the detector)
# ═══════════════════════════════════════════════════════════
@dataclass
class Config:
    # — Eye Aspect Ratio —
    ear_threshold: float = 0.22      # below = eyes closed
    closed_seconds: float = 3.0      # sustained closure → alarm

    # — PERCLOS (% frames eyes ≥ 80 % closed in last window_s seconds) —
    perclos_window: float = 60.0     # rolling window in seconds
    perclos_threshold: float = 0.35  # >35 % → drowsy flag

    # — Blink rate —
    blink_window: float = 60.0       # seconds
    blink_rate_thresh: float = 25.0  # blinks/min above this = drowsy

    # — Yawn detection —
    mar_threshold: float = 0.60      # Mouth Aspect Ratio for yawn
    yawn_min_duration: float = 1.2   # seconds mouth must stay open
    yawn_count_thresh: int = 3       # yawns in yawn_window → alarm
    yawn_window: float = 300.0       # 5 minutes

    # — Micro-sleep —
    micro_sleep_min: float = 0.5     # min seconds for a micro-sleep
    micro_sleep_max: float = 3.0     # max seconds (above = full closure alarm)

    # — Head nod —
    nod_pitch_thresh: float = 20.0   # degrees
    nod_count_thresh: int = 3
    nod_window: float = 30.0

    # — Look away —
    look_away_seconds: float = 10.0

    # — Drowsiness score weights (must sum to 1) —
    w_ear: float    = 0.30
    w_perclos: float = 0.30
    w_blink: float  = 0.15
    w_nod: float    = 0.15
    w_yawn: float   = 0.10

    # — Break reminder (Pomodoro) —
    break_interval_min: float = 25.0  # minutes
    break_duration_min: float = 5.0

    # — Alarm sound —
    alarm_file: str = None           # path to .wav/.mp3, or None → beep

    # — UI —
    sparkline_len: int = 180         # frames kept for EAR graph
    timeline_height: int = 8         # px for timeline strip

CFG = Config()

# ═══════════════════════════════════════════════════════════
#  MEDIAPIPE LANDMARK INDICES
# ═══════════════════════════════════════════════════════════
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 13, 14]
NOSE_TIP  = 1
CHIN      = 175
L_CHEEK   = 234
R_CHEEK   = 454
FOREHEAD  = 10

# ═══════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════
P = {
    "bg":       ( 10,  12,  20),   # very dark navy
    "panel":    ( 18,  22,  38),   # dark navy
    "panel2":   ( 24,  30,  50),   # slightly lighter panel
    "border":   ( 50,  80, 130),   # subtle blue border
    "accent":   ( 80, 170, 255),   # electric blue
    "accent2":  (130, 100, 255),   # purple accent
    "ok":       ( 60, 210, 130),   # emerald green
    "warn":     (255, 185,  50),   # amber
    "danger":   (240,  60,  60),   # red
    "danger2":  (255, 120,  60),   # orange-red
    "text":     (220, 230, 255),   # near-white blue
    "muted":    ( 90, 105, 145),   # dim blue-grey
    "spark":    ( 80, 200, 255),   # sparkline colour
    "focus_tl": ( 60, 210, 130),   # timeline focus
    "dist_tl":  (255, 185,  50),   # timeline distract
    "alarm_tl": (240,  60,  60),   # timeline alarm
}

# Source model (in the project folder, may have Unicode chars in path)
_SRC_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
# Safe ASCII copy used by MediaPipe C library (can't handle Unicode paths)
MODEL_PATH = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")),
                           "face_landmarker_safe.task")

# ═══════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═══════════════════════════════════════════════════════════

def ear(landmarks, indices, w, h):
    """Eye Aspect Ratio — lower → more closed."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)

def mar(landmarks, indices, w, h):
    """Mouth Aspect Ratio — higher → mouth open (yawn)."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    vert  = np.linalg.norm(pts[2] - pts[3])
    horiz = np.linalg.norm(pts[0] - pts[1])
    return vert / (horiz + 1e-6)

def get_pitch(landmarks, w, h):
    """Rough pitch in degrees — positive → looking down."""
    nose  = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    foreh = np.array([landmarks[FOREHEAD].x  * w, landmarks[FOREHEAD].y  * h])
    chin_ = np.array([landmarks[CHIN].x      * w, landmarks[CHIN].y      * h])
    mid   = (foreh + chin_) / 2
    dy    = nose[1] - mid[1]
    scale = np.linalg.norm(foreh - chin_) + 1e-6
    return math.degrees(math.asin(np.clip(dy / scale, -1, 1)))

def gaze_direction(landmarks, w, h):
    """True = face is roughly centred toward camera."""
    l_cheek = landmarks[L_CHEEK]
    r_cheek = landmarks[R_CHEEK]
    nose    = landmarks[NOSE_TIP]
    face_cx = (l_cheek.x + r_cheek.x) / 2
    return abs(nose.x - face_cx) < 0.12

# ═══════════════════════════════════════════════════════════
#  DROWSINESS SCORE  (0–100)
# ═══════════════════════════════════════════════════════════

def compute_drowsiness_score(ear_val, perclos, blink_rate, nod_count, yawn_count):
    """Weighted composite drowsiness score 0–100."""
    # Normalise each signal to [0, 1] (1 = maximally drowsy)
    ear_norm    = np.clip(1 - ear_val / max(CFG.ear_threshold, 0.01), 0, 1)
    pclos_norm  = np.clip(perclos / CFG.perclos_threshold, 0, 1)
    blink_norm  = np.clip(blink_rate / CFG.blink_rate_thresh, 0, 1)
    nod_norm    = np.clip(nod_count  / CFG.nod_count_thresh,  0, 1)
    yawn_norm   = np.clip(yawn_count / CFG.yawn_count_thresh, 0, 1)

    score = (
        CFG.w_ear    * ear_norm +
        CFG.w_perclos * pclos_norm +
        CFG.w_blink  * blink_norm +
        CFG.w_nod    * nod_norm +
        CFG.w_yawn   * yawn_norm
    )
    return float(np.clip(score * 100, 0, 100))

# ═══════════════════════════════════════════════════════════
#  ALARM MANAGER
# ═══════════════════════════════════════════════════════════

def _make_beep(freq=880, dur_ms=500):
    sr = 44100
    n  = int(sr * dur_ms / 1000)
    t  = np.linspace(0, dur_ms / 1000, n, endpoint=False)
    # Fade in/out to avoid clicks
    fade = np.ones(n)
    fade[:200]  = np.linspace(0, 1, 200)
    fade[-200:] = np.linspace(1, 0, 200)
    wave = (np.sin(2 * np.pi * freq * t) * 32767 * fade).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

class AlarmManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._sound   = None
        self._playing = False
        self.muted    = False

    def _load(self):
        if self._sound is None:
            if CFG.alarm_file and os.path.exists(CFG.alarm_file):
                self._sound = pygame.mixer.Sound(CFG.alarm_file)
            else:
                self._sound = _make_beep()

    def play(self):
        if self.muted:
            return
        if not self._playing:
            self._load()
            self._sound.play(loops=-1)
            self._playing = True

    def stop(self):
        if self._playing:
            self._sound.stop()
            self._playing = False

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted and self._playing:
            self.stop()

# ═══════════════════════════════════════════════════════════
#  SESSION / REPORT
# ═══════════════════════════════════════════════════════════

class Session:
    def __init__(self):
        self.start_time       = time.time()
        self.alarm_events     = []         # (timestamp, reason)
        self.blink_times      = []
        self.yawn_events      = []         # timestamps
        self.micro_sleeps     = []         # (start, end)
        self.focus_seconds    = 0.0
        self.distract_seconds = 0.0
        self._last_tick       = time.time()
        # Timeline: list of (timestamp, state) where state ∈ {0=focus,1=dist,2=alarm}
        self.timeline         = []
        self._break_count     = 0
        self._continuous_focus_start = time.time()
        self._perclos_buf     = deque()    # (timestamp, closed_bool)

    # ── tick ──────────────────────────────────────────────
    def tick(self, focused: bool, alarm_on: bool, eyes_closed: bool):
        now = time.time()
        dt  = now - self._last_tick
        self._last_tick = now

        if focused:
            self.focus_seconds += dt
        else:
            self.distract_seconds += dt
            self._continuous_focus_start = now

        state = 2 if alarm_on else (0 if focused else 1)
        if not self.timeline or self.timeline[-1][1] != state:
            self.timeline.append((now, state))

        # PERCLOS buffer
        self._perclos_buf.append((now, eyes_closed))
        cutoff = now - CFG.perclos_window
        while self._perclos_buf and self._perclos_buf[0][0] < cutoff:
            self._perclos_buf.popleft()

    def perclos(self):
        if not self._perclos_buf:
            return 0.0
        closed = sum(1 for _, c in self._perclos_buf if c)
        return closed / len(self._perclos_buf)

    def log_alarm(self, reason: str):
        self.alarm_events.append((time.time(), reason))

    def log_blink(self):
        self.blink_times.append(time.time())

    def log_yawn(self):
        self.yawn_events.append(time.time())

    def log_micro_sleep(self, start, end):
        self.micro_sleeps.append((start, end))

    def blink_rate(self):
        now    = time.time()
        recent = [t for t in self.blink_times if now - t < CFG.blink_window]
        return len(recent) / (CFG.blink_window / 60)

    def yawn_count_recent(self):
        now = time.time()
        return sum(1 for t in self.yawn_events if now - t < CFG.yawn_window)

    def duration(self):
        return time.time() - self.start_time

    def continuous_focus_min(self):
        return (time.time() - self._continuous_focus_start) / 60

    def log_break(self):
        self._break_count += 1
        self._continuous_focus_start = time.time()

    def save_report(self):
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"focus_report_{ts}.json"
        dur   = self.duration()
        data  = {
            "date":             datetime.datetime.now().isoformat(),
            "duration_min":     round(dur / 60, 2),
            "focus_min":        round(self.focus_seconds / 60, 2),
            "distracted_min":   round(self.distract_seconds / 60, 2),
            "focus_pct":        round(100 * self.focus_seconds / max(dur, 1), 1),
            "total_alarms":     len(self.alarm_events),
            "alarm_log":        [{"time": t, "reason": r} for t, r in self.alarm_events],
            "avg_blink_rate":   round(self.blink_rate(), 1),
            "perclos_avg_pct":  round(self.perclos() * 100, 1),
            "yawn_count":       len(self.yawn_events),
            "micro_sleep_count": len(self.micro_sleeps),
            "micro_sleep_durations": [round(e - s, 2) for s, e in self.micro_sleeps],
            "break_count":      self._break_count,
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        return fname, data

# ═══════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════

def _alpha_rect(frame, x, y, w, h, color, alpha):
    """Blend a filled rectangle onto frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_panel(frame, x, y, w, h, color=None, alpha=0.70, border=True,
               radius=8, border_color=None):
    """Rounded-ish panel with optional border."""
    c = color or P["panel"]
    _alpha_rect(frame, x, y, w, h, c, alpha)
    if border:
        bc = border_color or P["border"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), bc, 1, cv2.LINE_AA)

def put_text(frame, text, pos, scale=0.55, color=None, thickness=1,
             bold=False, shadow=True):
    c    = color or P["text"]
    font = cv2.FONT_HERSHEY_DUPLEX
    if shadow:
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
    if bold:
        cv2.putText(frame, text, pos, font, scale, (0,0,0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, c, thickness, cv2.LINE_AA)

def draw_bar(frame, x, y, w, h, value, color, bg=None, border=True):
    """Horizontal progress bar."""
    bg_c = bg or P["panel2"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_c, -1)
    if border:
        cv2.rectangle(frame, (x, y), (x + w, y + h), P["border"], 1)
    fill = int(w * np.clip(value, 0, 1))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)

def draw_score_bar(frame, x, y, w, h, score):
    """Gradient score bar 0-100."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["panel2"], -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["border"], 1)
    fill = int(w * score / 100)
    if fill > 0:
        # gradient green→yellow→red
        for i in range(fill):
            t   = i / max(w, 1)
            if t < 0.5:
                r = int(60  + t * 2 * (255 - 60))
                g = int(210 - t * 2 * (210 - 185))
                b = int(130 - t * 2 * 130)
            else:
                r = 255
                g = int(185 - (t - 0.5) * 2 * (185 - 60))
                b = 60
            cv2.line(frame, (x + i, y), (x + i, y + h), (b, g, r))

def draw_eye_outline(frame, landmarks, indices, w, h, color, thickness=1):
    pts = np.array([[int(landmarks[i].x*w), int(landmarks[i].y*h)]
                    for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    # Pupil dot
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

def draw_sparkline(frame, x, y, w, h, values, color, min_v=0.0, max_v=0.5):
    """Mini line chart."""
    if len(values) < 2:
        return
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["panel2"], -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["border"], 1)
    # Threshold line
    thresh_y = int(y + h * (1 - (CFG.ear_threshold - min_v) / (max_v - min_v + 1e-6)))
    thresh_y = np.clip(thresh_y, y, y + h)
    cv2.line(frame, (x, thresh_y), (x + w, thresh_y), P["danger"], 1, cv2.LINE_AA)

    n  = len(values)
    xs = np.linspace(x, x + w, n).astype(int)
    ys = [int(y + h * (1 - (v - min_v) / (max_v - min_v + 1e-6))) for v in values]
    ys = [np.clip(yy, y, y + h) for yy in ys]
    for i in range(1, n):
        cv2.line(frame, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 1, cv2.LINE_AA)

    # Label
    put_text(frame, "EAR", (x + 3, y + 11), scale=0.30, color=P["muted"], shadow=False)
    put_text(frame, f"{values[-1]:.2f}", (x + w - 32, y + 11), scale=0.30,
             color=color, shadow=False)

def draw_timeline_strip(frame, x, y, w, h, timeline, session_start, now):
    """Coloured timeline bar at bottom."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["panel2"], -1)
    dur = now - session_start
    if dur < 1:
        return
    state_colors = [P["focus_tl"], P["dist_tl"], P["alarm_tl"]]
    for i, (ts, state) in enumerate(timeline):
        t_end   = timeline[i+1][0] if i+1 < len(timeline) else now
        px_s    = x + int(w * (ts - session_start) / dur)
        px_e    = x + int(w * (t_end - session_start) / dur)
        if px_e > px_s:
            cv2.rectangle(frame, (px_s, y), (px_e, y + h), state_colors[state], -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), P["border"], 1)

def draw_pulse_ring(frame, cx, cy, radius, t, color):
    """Animated pulsing ring — t=time.time()."""
    pulse = abs(math.sin(t * 4))
    r_dyn = int(radius + 10 * pulse)
    alpha_val = int(80 + 175 * pulse)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r_dyn, color, 3, cv2.LINE_AA)
    mask = np.zeros_like(frame)
    cv2.circle(mask, (cx, cy), r_dyn, (alpha_val, alpha_val, alpha_val), 3, cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, pulse * 0.9, frame, 1 - pulse * 0.9, 0)
    frame[:] = blended

def score_color(score):
    if score < 40:
        return P["ok"]
    elif score < 70:
        return P["warn"]
    return P["danger"]

def hms(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ═══════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════

def download_model():
    """Ensure face_landmarker.task exists at a safe ASCII path for MediaPipe."""
    import shutil
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    # If the source model exists in the project folder, copy it to safe path
    if os.path.exists(_SRC_MODEL):
        if not os.path.exists(MODEL_PATH):
            print("[Sleep Alarm] Copying model to safe temp path...")
            shutil.copy2(_SRC_MODEL, MODEL_PATH)
            print(f"[Sleep Alarm] Model ready at: {MODEL_PATH}")
        return

    # Otherwise download directly to the safe path
    if not os.path.exists(MODEL_PATH):
        print("[Sleep Alarm] Downloading face landmark model (~6 MB)...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"[Sleep Alarm] Model downloaded to: {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    download_model()

    # ── MediaPipe setup ──────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_mesh = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    alarm   = AlarmManager()
    session = Session()

    # ── State ────────────────────────────────────────────
    eyes_closed_since  = None
    look_away_since    = None
    nod_times          = []
    last_ear           = 1.0
    blink_in_progress  = False
    alarm_active       = False
    alarm_reason       = ""
    face_detected      = False
    ear_history        = deque(maxlen=CFG.sparkline_len)

    # Yawn state
    yawn_open_since    = None

    # Micro-sleep state
    micro_sleep_start  = None

    # Alarm dismiss: hold SPACE
    space_held_since   = None
    SPACE_HOLD_SECS    = 1.0

    # Break
    break_enabled      = True
    break_alert_active = False

    WIN = "Sleep Alarm v3 — Study Assistant"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    print("\n[Sleep Alarm v3]  Q=quit  R=report  S=mute  B=break  +/-=EAR\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = face_mesh.detect(mp_img)

        now          = time.time()
        face_detected = len(result.face_landmarks) > 0

        # ── Subtle dark tint over raw frame ──────────────
        overlay0 = frame.copy()
        cv2.rectangle(overlay0, (0, 0), (W, H), P["bg"], -1)
        cv2.addWeighted(overlay0, 0.22, frame, 0.78, 0, frame)

        current_ear      = last_ear
        current_mar      = 0.0
        looking_at_screen = True
        pitch            = 0.0
        eyes_really_closed = False

        if face_detected:
            lm = result.face_landmarks[0]

            # ── Metrics ───────────────────────────────────
            left_ear       = ear(lm, LEFT_EYE,  W, H)
            right_ear      = ear(lm, RIGHT_EYE, W, H)
            current_ear    = (left_ear + right_ear) / 2
            current_mar    = mar(lm, MOUTH, W, H)
            looking_at_screen = gaze_direction(lm, W, H)
            pitch          = get_pitch(lm, W, H)
            eyes_really_closed = current_ear < CFG.ear_threshold

            # ── Blink ─────────────────────────────────────
            if current_ear < CFG.ear_threshold and last_ear >= CFG.ear_threshold:
                blink_in_progress = True
            elif current_ear >= CFG.ear_threshold and blink_in_progress:
                session.log_blink()
                blink_in_progress = False
            last_ear = current_ear

            # ── Eye close timer + micro-sleep ────────────
            if eyes_really_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                    micro_sleep_start = now
            else:
                if eyes_closed_since is not None:
                    dur_closed = now - eyes_closed_since
                    if CFG.micro_sleep_min <= dur_closed < CFG.micro_sleep_max:
                        session.log_micro_sleep(eyes_closed_since, now)
                eyes_closed_since = None

            # ── Yawn ──────────────────────────────────────
            if current_mar >= CFG.mar_threshold:
                if yawn_open_since is None:
                    yawn_open_since = now
            else:
                if yawn_open_since is not None:
                    if now - yawn_open_since >= CFG.yawn_min_duration:
                        session.log_yawn()
                    yawn_open_since = None

            # ── Head nod ──────────────────────────────────
            if pitch > CFG.nod_pitch_thresh:
                if not nod_times or (now - nod_times[-1]) > 1.5:
                    nod_times.append(now)
            nod_times = [t for t in nod_times if now - t < CFG.nod_window]

            # ── Look-away ─────────────────────────────────
            if not looking_at_screen:
                if look_away_since is None:
                    look_away_since = now
            else:
                look_away_since = None

            session.tick(
                focused      = looking_at_screen and not eyes_really_closed,
                alarm_on     = alarm_active,
                eyes_closed  = eyes_really_closed,
            )

            # ── Eye landmarks ─────────────────────────────
            eye_col = P["danger"] if eyes_really_closed else P["accent"]
            lthick  = 2 if eyes_really_closed else 1
            draw_eye_outline(frame, lm, LEFT_EYE,  W, H, eye_col, lthick)
            draw_eye_outline(frame, lm, RIGHT_EYE, W, H, eye_col, lthick)

        else:
            session.tick(focused=False, alarm_on=alarm_active, eyes_closed=False)
            eyes_closed_since = None
            look_away_since   = None

        ear_history.append(current_ear)

        # ── PERCLOS + derived ─────────────────────────────
        perclos      = session.perclos()
        brate        = session.blink_rate()
        yawn_recent  = session.yawn_count_recent()
        d_score      = compute_drowsiness_score(current_ear, perclos, brate,
                                                len(nod_times), yawn_recent)

        # ─────────────────────────────────────────────────
        #  ALARM TRIGGER
        # ─────────────────────────────────────────────────
        new_alarm = False
        reason    = ""

        if eyes_closed_since and (now - eyes_closed_since) >= CFG.closed_seconds:
            new_alarm = True
            reason    = f"Eyes closed {now - eyes_closed_since:.1f}s!"

        elif len(nod_times) >= CFG.nod_count_thresh:
            new_alarm = True
            reason    = f"Head nodding ({len(nod_times)}x)"

        elif brate > CFG.blink_rate_thresh:
            new_alarm = True
            reason    = f"High blink rate ({brate:.0f}/min)"

        elif look_away_since and (now - look_away_since) >= CFG.look_away_seconds:
            new_alarm = True
            reason    = f"Not looking at screen ({now - look_away_since:.0f}s)"

        elif perclos >= CFG.perclos_threshold:
            new_alarm = True
            reason    = f"PERCLOS {perclos*100:.0f}% — drowsy!"

        elif yawn_recent >= CFG.yawn_count_thresh:
            new_alarm = True
            reason    = f"Frequent yawning ({yawn_recent}x)"

        if new_alarm and not alarm_active:
            alarm_active = True
            alarm_reason = reason
            alarm.play()
            session.log_alarm(reason)
        elif not new_alarm and alarm_active:
            alarm_active = False
            alarm_reason = ""
            alarm.stop()

        # Break reminder
        if break_enabled:
            cf_min = session.continuous_focus_min()
            break_alert_active = cf_min >= CFG.break_interval_min
        else:
            break_alert_active = False

        # ─────────────────────────────────────────────────
        #  UI — BACKGROUND GRADIENT OVERLAY
        # ─────────────────────────────────────────────────
        # Subtle vignette
        vig = np.zeros((H, W, 3), dtype=np.uint8)
        for yy in range(H):
            alpha_v = int(40 * (1 - abs(yy - H/2) / (H/2)))
            vig[yy, :] = (alpha_v, alpha_v, alpha_v)
        cv2.addWeighted(vig, 0.3, frame, 1.0, 0, frame)

        dur      = session.duration()
        fp       = session.focus_seconds / max(dur, 1)

        # ─────────────────────────────────────────────────
        #  LEFT PANEL  (metrics + sparkline)
        # ─────────────────────────────────────────────────
        LP_X, LP_Y, LP_W, LP_H = 14, 14, 270, 410
        draw_panel(frame, LP_X, LP_Y, LP_W, LP_H, alpha=0.80)

        # Header
        put_text(frame, "STUDY ASSISTANT", (LP_X+12, LP_Y+26),
                 scale=0.55, color=P["accent"], thickness=1)
        put_text(frame, "Drowsiness Monitor", (LP_X+12, LP_Y+44),
                 scale=0.36, color=P["muted"])

        # Divider
        cv2.line(frame, (LP_X+8, LP_Y+52), (LP_X+LP_W-8, LP_Y+52),
                 P["border"], 1, cv2.LINE_AA)

        yo = LP_Y + 66
        # Session timer
        put_text(frame, "SESSION", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, hms(dur), (LP_X+LP_W-80, yo), scale=0.46, color=P["text"])
        yo += 24

        # Focus %
        fc_col = P["ok"] if fp > 0.7 else P["warn"] if fp > 0.4 else P["danger"]
        put_text(frame, "FOCUS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{fp*100:.0f}%", (LP_X+LP_W-52, yo), scale=0.46, color=fc_col)
        yo += 14
        draw_bar(frame, LP_X+12, yo, LP_W-24, 7, fp, fc_col)
        yo += 20

        # EAR
        ec = P["ok"] if current_ear >= CFG.ear_threshold else P["danger"]
        put_text(frame, "EAR", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{current_ear:.3f}", (LP_X+LP_W-60, yo), scale=0.46, color=ec)
        yo += 14
        draw_bar(frame, LP_X+12, yo, LP_W-24, 7, current_ear / 0.45, ec)
        yo += 20

        # PERCLOS
        pc_col = P["ok"] if perclos < 0.2 else P["warn"] if perclos < CFG.perclos_threshold else P["danger"]
        put_text(frame, "PERCLOS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{perclos*100:.0f}%", (LP_X+LP_W-60, yo), scale=0.46, color=pc_col)
        yo += 14
        draw_bar(frame, LP_X+12, yo, LP_W-24, 7, perclos / CFG.perclos_threshold, pc_col)
        yo += 20

        # Blink rate
        br_col = P["ok"] if brate < CFG.blink_rate_thresh*0.7 else P["warn"] if brate < CFG.blink_rate_thresh else P["danger"]
        put_text(frame, "BLINKS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{brate:.0f}/min", (LP_X+LP_W-72, yo), scale=0.46, color=br_col)
        yo += 20

        # Nods
        nc = P["ok"] if len(nod_times) < CFG.nod_count_thresh else P["danger"]
        put_text(frame, "NODS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{len(nod_times)}/{CFG.nod_count_thresh}", (LP_X+LP_W-60, yo), scale=0.46, color=nc)
        yo += 20

        # Yawns
        yc = P["ok"] if yawn_recent < CFG.yawn_count_thresh else P["danger"]
        put_text(frame, "YAWNS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{yawn_recent}", (LP_X+LP_W-48, yo), scale=0.46, color=yc)
        yo += 20

        # Micro-sleeps
        ms_count = len(session.micro_sleeps)
        ms_col   = P["ok"] if ms_count == 0 else P["warn"] if ms_count < 3 else P["danger"]
        put_text(frame, "μ-SLEEP", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{ms_count}", (LP_X+LP_W-48, yo), scale=0.46, color=ms_col)
        yo += 20

        # Alarms
        ac = P["ok"] if not session.alarm_events else P["danger"]
        put_text(frame, "ALARMS", (LP_X+12, yo), scale=0.36, color=P["muted"])
        put_text(frame, f"{len(session.alarm_events)}", (LP_X+LP_W-48, yo), scale=0.46, color=ac)
        yo += 18

        # EAR sparkline
        cv2.line(frame, (LP_X+8, yo), (LP_X+LP_W-8, yo), P["border"], 1)
        yo += 8
        draw_sparkline(frame, LP_X+12, yo, LP_W-24, 48, list(ear_history),
                       P["spark"])

        # ─────────────────────────────────────────────────
        #  DROWSINESS SCORE  (bottom-left of left panel area)
        # ─────────────────────────────────────────────────
        SCR_Y = LP_Y + LP_H + 10
        draw_panel(frame, LP_X, SCR_Y, LP_W, 56, alpha=0.80)
        sc_col = score_color(d_score)
        put_text(frame, "DROWSINESS SCORE", (LP_X+12, SCR_Y+17),
                 scale=0.38, color=P["muted"])
        put_text(frame, f"{d_score:.0f}", (LP_X+LP_W-46, SCR_Y+38),
                 scale=0.70, color=sc_col, thickness=2)
        draw_score_bar(frame, LP_X+12, SCR_Y+26, LP_W-60, 10, d_score)

        # ─────────────────────────────────────────────────
        #  RIGHT PANEL  (status + clock + break)
        # ─────────────────────────────────────────────────
        RP_W, RP_H = 200, 200
        RP_X = W - RP_W - 14
        RP_Y = 14
        draw_panel(frame, RP_X, RP_Y, RP_W, RP_H, alpha=0.80)

        # Status badge
        if not face_detected:
            status_label = "NO FACE"
            status_col   = P["muted"]
        elif alarm_active:
            status_label = "DROWSY!"
            status_col   = P["danger"]
        else:
            status_label = "FOCUSED"
            status_col   = P["ok"]

        # Pulsing status dot
        pulse_r = int(6 + 3 * abs(math.sin(now * 3)))
        cv2.circle(frame, (RP_X + 20, RP_Y + 26), pulse_r, status_col, -1, cv2.LINE_AA)
        put_text(frame, status_label, (RP_X+32, RP_Y+32), scale=0.62,
                 color=status_col, thickness=2, bold=True)

        # Divider
        cv2.line(frame, (RP_X+8, RP_Y+44), (RP_X+RP_W-8, RP_Y+44),
                 P["border"], 1, cv2.LINE_AA)

        # Clock
        clock_str = datetime.datetime.now().strftime("%H:%M:%S")
        put_text(frame, clock_str, (RP_X+18, RP_Y+68), scale=0.58, color=P["accent"])
        date_str  = datetime.datetime.now().strftime("%d %b %Y")
        put_text(frame, date_str,  (RP_X+18, RP_Y+88), scale=0.36, color=P["muted"])

        # Gaze
        gaze_txt = "On screen" if looking_at_screen else "Looking away"
        gaze_col = P["ok"] if looking_at_screen else P["warn"]
        put_text(frame, gaze_txt, (RP_X+12, RP_Y+112), scale=0.40, color=gaze_col)

        # Pitch
        put_text(frame, f"Pitch {pitch:+.1f}°", (RP_X+12, RP_Y+130), scale=0.38, color=P["muted"])

        # EAR threshold
        put_text(frame, f"EAR thr {CFG.ear_threshold:.2f}", (RP_X+12, RP_Y+148),
                 scale=0.38, color=P["muted"])

        # Mute indicator
        if alarm.muted:
            put_text(frame, "[MUTED]", (RP_X+12, RP_Y+166), scale=0.42, color=P["warn"])

        # Break countdown
        if break_enabled:
            remaining_break = max(0, CFG.break_interval_min - session.continuous_focus_min())
            bc_col = P["danger"] if break_alert_active else P["muted"]
            br_label = "BREAK NOW!" if break_alert_active else f"Break in {remaining_break:.0f}m"
            put_text(frame, br_label, (RP_X+12, RP_Y+186), scale=0.40, color=bc_col)

        # ─────────────────────────────────────────────────
        #  RIGHT-BOTTOM: Hotkeys legend
        # ─────────────────────────────────────────────────
        KP_Y = RP_Y + RP_H + 10
        KP_H = 108
        draw_panel(frame, RP_X, KP_Y, RP_W, KP_H, alpha=0.75)
        keys = [("Q", "Quit"), ("R", "Save report"),
                ("S", "Mute toggle"), ("B", "Break timer"),
                ("+/-", "EAR thresh"), ("SPC", "Dismiss alarm")]
        for ki, (k, desc) in enumerate(keys):
            ky = KP_Y + 18 + ki * 16
            put_text(frame, k, (RP_X+12, ky), scale=0.34, color=P["accent2"])
            put_text(frame, desc, (RP_X+44, ky), scale=0.34, color=P["muted"])

        # ─────────────────────────────────────────────────
        #  CENTRE OVERLAYS
        # ─────────────────────────────────────────────────

        # ── Eye-close countdown ring ──────────────────────
        if eyes_closed_since:
            elapsed = now - eyes_closed_since
            remain  = max(0, CFG.closed_seconds - elapsed)
            cx, cy  = W // 2, H // 2
            frac    = elapsed / CFG.closed_seconds
            ring_r  = 64
            # Outer glow
            _alpha_rect(frame, cx-ring_r-12, cy-ring_r-12,
                        (ring_r+12)*2, (ring_r+12)*2, P["danger"], 0.08)
            cv2.circle(frame, (cx, cy), ring_r, P["panel"], 2)
            cv2.ellipse(frame, (cx, cy), (ring_r, ring_r), -90,
                        0, int(360 * frac), P["danger"], 4, cv2.LINE_AA)
            put_text(frame, f"{remain:.1f}s", (cx-22, cy+8), scale=0.82,
                     color=P["danger"], thickness=2, bold=True)
            put_text(frame, "Eyes closing...", (cx-72, cy+38), scale=0.46,
                     color=P["warn"])

        # ── ALARM BANNER ──────────────────────────────────
        if alarm_active:
            # Animated pulse rings
            for ring_off, ring_alpha in [(0, 0.5), (20, 0.3), (40, 0.15)]:
                pulse_v = abs(math.sin(now * 5 + ring_off * 0.1))
                ov_ring = frame.copy()
                cv2.rectangle(ov_ring, (0, 0), (W, H), P["danger"], ring_off + 2)
                cv2.addWeighted(ov_ring, ring_alpha * pulse_v, frame,
                                1 - ring_alpha * pulse_v, 0, frame)

            bh = 80
            by = H // 2 + 70
            # Banner background
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, by), (W, by + bh), (100, 0, 0), -1)
            cv2.addWeighted(ov2, 0.80, frame, 0.20, 0, frame)
            # Flicker border
            if int(now * 5) % 2 == 0:
                cv2.rectangle(frame, (0, by), (W, by + bh), P["danger"], 3)

            txt_w = cv2.getTextSize("!! WAKE UP !!", cv2.FONT_HERSHEY_DUPLEX,
                                       1.0, 2)[0][0]
            put_text(frame, "!! WAKE UP !!", (W//2 - txt_w//2, by+42), scale=1.0,
                     color=(255, 255, 255), thickness=2, bold=True, shadow=True)

            reason_w = cv2.getTextSize(alarm_reason, cv2.FONT_HERSHEY_DUPLEX,
                                          0.48, 1)[0][0]
            put_text(frame, alarm_reason, (W//2 - reason_w//2, by+66),
                     scale=0.48, color=P["warn"])

            # SPACE hold progress
            if space_held_since:
                hold_frac = min(1.0, (now - space_held_since) / SPACE_HOLD_SECS)
                bar_w = 200
                bar_x = W//2 - bar_w//2
                draw_bar(frame, bar_x, by - 20, bar_w, 8, hold_frac, P["ok"])
                put_text(frame, "Hold SPACE to dismiss",
                         (bar_x - 8, by - 26), scale=0.36, color=P["muted"])

        # ── Break banner ──────────────────────────────────
        if break_alert_active and not alarm_active:
            bh = 50
            by = H // 2 - 80
            ov3 = frame.copy()
            cv2.rectangle(ov3, (0, by), (W, by + bh), (20, 60, 20), -1)
            cv2.addWeighted(ov3, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (0, by), (W, by + bh), P["ok"], 2)
            put_text(frame, "Time for a break! Press B to snooze.",
                     (W//2 - 185, by + 32), scale=0.58, color=P["ok"], bold=True)

        # ─────────────────────────────────────────────────
        #  BOTTOM: timeline + status bar
        # ─────────────────────────────────────────────────
        TL_H  = CFG.timeline_height
        TL_Y  = H - TL_H - 28
        draw_timeline_strip(frame, 0, TL_Y, W, TL_H,
                            session.timeline, session.start_time, now)

        # Status bar
        sb_y = H - 22
        _alpha_rect(frame, 0, sb_y, W, 22, P["panel"], 0.80)
        hint = ("Q=Quit  R=Report  S=Mute  B=Break  +/-=EAR  "
                "Hold SPACE=Dismiss alarm")
        put_text(frame, hint, (12, H - 7), scale=0.34, color=P["muted"], shadow=False)

        # ─────────────────────────────────────────────────
        #  FRAME CORNER ACCENTS
        # ─────────────────────────────────────────────────
        ca_len = 22
        ca_col = P["accent"] if not alarm_active else P["danger"]
        ca_t   = 2
        for cx_, cy_, dx, dy in [(0,0,1,1),(W,0,-1,1),(0,H,1,-1),(W,H,-1,-1)]:
            cv2.line(frame, (cx_, cy_), (cx_+dx*ca_len, cy_), ca_col, ca_t)
            cv2.line(frame, (cx_, cy_), (cx_, cy_+dy*ca_len), ca_col, ca_t)

        # ─────────────────────────────────────────────────
        #  SHOW
        # ─────────────────────────────────────────────────
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────

        # SPACE held → alarm dismiss
        if key == ord(' '):
            if alarm_active:
                if space_held_since is None:
                    space_held_since = now
                elif now - space_held_since >= SPACE_HOLD_SECS:
                    alarm.stop()
                    alarm_active      = False
                    alarm_reason      = ""
                    eyes_closed_since = None
                    nod_times         = []
                    space_held_since  = None
        else:
            space_held_since = None

        if key == ord('q') or key == 27:
            break

        if key == ord('r'):
            fname, data = session.save_report()
            sep = "+" + "-"*36 + "+"
            print(f"\n{sep}")
            print(f"|{'SESSION REPORT SAVED':^36}|")
            print(f"{sep}")
            print(f"|  File:         {fname}")
            print(f"|  Duration:     {data['duration_min']} min")
            print(f"|  Focus:        {data['focus_pct']}%  ({data['focus_min']} min)")
            print(f"|  Alarms:       {data['total_alarms']}")
            print(f"|  Blink rate:   {data['avg_blink_rate']}/min")
            print(f"|  PERCLOS avg:  {data['perclos_avg_pct']}%")
            print(f"|  Yawns:        {data['yawn_count']}")
            print(f"|  Micro-sleeps: {data['micro_sleep_count']}")
            print(f"|  Breaks taken: {data['break_count']}")
            print(f"{sep}\n")

        if key == ord('s'):
            alarm.toggle_mute()
            print(f"[Mute] {'ON' if alarm.muted else 'OFF'}")

        if key == ord('b'):
            if break_alert_active:
                # Snooze break
                session.log_break()
                break_alert_active = False
                print("[Break] Snoozed — timer reset.")
            else:
                break_enabled = not break_enabled
                print(f"[Break reminder] {'ON' if break_enabled else 'OFF'}")

        if key == ord('+') or key == ord('='):
            CFG.ear_threshold = min(CFG.ear_threshold + 0.01, 0.40)
            print(f"[EAR threshold] -> {CFG.ear_threshold:.2f}")

        if key == ord('-') or key == ord('_'):
            CFG.ear_threshold = max(CFG.ear_threshold - 0.01, 0.10)
            print(f"[EAR threshold] -> {CFG.ear_threshold:.2f}")

    # ── Cleanup ───────────────────────────────────────────
    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    try:
        face_mesh.__exit__(None, None, None)
    except Exception:
        pass

    print("\n[Sleep Alarm v3] Session ended.")
    fname, data = session.save_report()
    print(f"\n╔══════════════════════════════════╗")
    print(f"║       FINAL SESSION REPORT       ║")
    print(f"╠══════════════════════════════════╣")
    print(f"║  File:         {fname}")
    print(f"║  Duration:     {data['duration_min']} min")
    print(f"║  Focus:        {data['focus_pct']}%  ({data['focus_min']} min)")
    print(f"║  Distracted:   {data['distracted_min']} min")
    print(f"║  Alarms:       {data['total_alarms']}")
    print(f"║  Blink rate:   {data['avg_blink_rate']}/min")
    print(f"║  PERCLOS avg:  {data['perclos_avg_pct']}%")
    print(f"║  Yawns:        {data['yawn_count']}")
    print(f"║  Micro-sleeps: {data['micro_sleep_count']}")
    print(f"║  Breaks taken: {data['break_count']}")
    print(f"╚══════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
