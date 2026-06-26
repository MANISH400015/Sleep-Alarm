# 😴 Sleep Alarm v3 — AI Drowsiness & Study Monitor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6D00)
![pygame](https://img.shields.io/badge/pygame-2.5%2B-1A1A2E)
![License](https://img.shields.io/badge/License-MIT-22C55E)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-6366F1)

**Real-time webcam-based drowsiness detector that alerts you before you fall asleep at your desk.**  
Uses Google MediaPipe's Face Landmarker (478 points) to track eyes, head pose, yawns, and more.

</div>

---

## 📸 What It Looks Like

The app opens a live webcam window with a full dark-mode HUD overlay:

- **Left panel** — EAR, PERCLOS, blink rate, nod count, yawn count, micro-sleeps, EAR sparkline graph
- **Bottom-left** — Drowsiness Score 0–100 (gradient green → red bar)
- **Right panel** — Status badge, live clock, gaze direction, break countdown, hotkey legend
- **Bottom strip** — Full session timeline (green = focused, amber = distracted, red = alarm)
- **Centre overlay** — Countdown ring when eyes are closing; pulsing alarm banner when triggered

---

## ✨ Detection Features

| Signal | Trigger |
|--------|---------|
| **Eye closure (EAR)** | Eyes closed > 3 seconds → alarm |
| **PERCLOS** | >35% frames eyes closed in 60 s → drowsy |
| **Micro-sleep** | 0.5–3 s eye closures logged separately |
| **Blink rate** | >25 blinks/min → fatigue alarm |
| **Head nod** | 3+ nods in 30 s → alarm |
| **Yawn rate** | 3+ yawns in 5 min → alarm |
| **Look-away** | Not looking at screen for >10 s → alert |
| **Pomodoro break** | 25 min continuous focus → break reminder |

All thresholds are configurable at the top of the script.

---

## 🛠️ Installation — Step by Step

### Step 1 — Check Python version

You need **Python 3.9 or newer**. Open a terminal and run:

```bash
python --version
# Should print: Python 3.9.x  or higher
```

If Python is not installed, download it from 👉 https://www.python.org/downloads/

> **Windows users:** During installation, check ✅ **"Add Python to PATH"**

---

### Step 2 — Clone or download this repo

**Option A — Clone with Git (recommended):**
```bash
git clone https://github.com/YOUR_USERNAME/sleep-alarm.git
cd sleep-alarm
```

**Option B — Download ZIP:**
1. Click the green **Code** button on this page → **Download ZIP**
2. Extract the ZIP
3. Open a terminal inside the extracted folder

---

### Step 3 — (Recommended) Create a virtual environment

This keeps the project's packages isolated from your system Python.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal prompt.

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | What it does |
|---------|---------|--------------|
| `opencv-python` | ≥ 4.8 | Webcam capture, frame rendering, HUD drawing |
| `mediapipe` | ≥ 0.10 | Face landmark detection (478 landmarks per face) |
| `numpy` | ≥ 1.24 | Numerical math for EAR / MAR / pitch calculations |
| `pygame` | ≥ 2.5 | Alarm beep sound playback |

> ⏱️ This may take 1–3 minutes depending on your internet speed (MediaPipe is ~50 MB).

---

### Step 5 — Run the app

```bash
python sleep_alarm2.py
```

**First launch only:** The app will automatically download the MediaPipe face landmark model (~6 MB) and save it next to the script. This only happens once.

A window titled **"Sleep Alarm v3 — Study Assistant"** will open showing your webcam feed with the full HUD overlay.

---

## ⌨️ Hotkeys (while the app is running)

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit the app (auto-saves session report) |
| `R` | Save session report to JSON right now |
| `S` | Toggle alarm sound mute / unmute |
| `B` | Toggle Pomodoro break reminder on/off (or snooze if break alert is active) |
| `+` or `=` | Increase EAR threshold by 0.01 (less sensitive) |
| `-` | Decrease EAR threshold by 0.01 (more sensitive) |
| Hold `Space` (1 sec) | Dismiss the active alarm |

---

## ⚙️ Configuration

Open `sleep_alarm2.py` and edit the `Config` dataclass near the top of the file:

```python
@dataclass
class Config:
    # ── Eye detection ──────────────────────────────────────
    ear_threshold: float = 0.22      # EAR below this = eyes closed
                                     # Lower = more sensitive; typical range 0.15–0.30
    closed_seconds: float = 3.0      # Seconds eyes must stay closed before alarm fires

    # ── PERCLOS ────────────────────────────────────────────
    perclos_window: float = 60.0     # Rolling window in seconds
    perclos_threshold: float = 0.35  # >35% eyes-closed frames = drowsy

    # ── Blink rate ─────────────────────────────────────────
    blink_rate_thresh: float = 25.0  # Blinks/min above this = drowsy alarm

    # ── Yawn ───────────────────────────────────────────────
    mar_threshold: float = 0.60      # Mouth Aspect Ratio — higher = more open
    yawn_min_duration: float = 1.2   # Mouth must stay open this long to count
    yawn_count_thresh: int = 3       # Yawns within yawn_window → alarm
    yawn_window: float = 300.0       # 5-minute window for yawn counting

    # ── Head nod ───────────────────────────────────────────
    nod_pitch_thresh: float = 20.0   # Degrees nose-down before counting as a nod
    nod_count_thresh: int = 3        # Nods in nod_window → alarm
    nod_window: float = 30.0         # 30-second nod window

    # ── Look away ──────────────────────────────────────────
    look_away_seconds: float = 10.0  # Seconds looking away → alert

    # ── Pomodoro break ─────────────────────────────────────
    break_interval_min: float = 25.0 # Focus interval before break reminder (minutes)
    break_duration_min: float = 5.0  # Suggested break duration

    # ── Custom alarm sound ─────────────────────────────────
    alarm_file: str = None           # Path to a .wav or .mp3 file
                                     # Set to None to use the built-in beep
```

---

## 📊 Session Reports

Every time you quit (or press `R`), a JSON report is saved in the same folder:

```
focus_report_20260626_012626.json
```

Example contents:
```json
{
  "date": "2026-06-26T01:26:26",
  "duration_min": 45.2,
  "focus_min": 37.3,
  "distracted_min": 7.9,
  "focus_pct": 82.5,
  "total_alarms": 3,
  "alarm_log": [
    {"time": 1782417120.4, "reason": "Eyes closed 3.2s!"},
    {"time": 1782417480.1, "reason": "PERCLOS 38% - drowsy!"}
  ],
  "avg_blink_rate": 18.4,
  "perclos_avg_pct": 12.4,
  "yawn_count": 2,
  "micro_sleep_count": 5,
  "micro_sleep_durations": [0.62, 0.81, 1.1, 0.55, 0.73],
  "break_count": 1
}
```

---

## 🐛 Troubleshooting

### Webcam not detected
```
ERROR: Could not open webcam.
```
- Make sure no other app (Teams, Zoom, OBS) is using the camera
- Try a different camera index: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the script

### Face not detected / EAR always 0
- Improve lighting — face the light source, avoid backlight
- Move closer to the camera (stay within ~1 metre)
- Ensure glasses/masks don't cover the eye area heavily

### Too many false alarms
- Lower `ear_threshold` (e.g. `0.18`) so eyes need to be more closed to trigger
- Raise `closed_seconds` (e.g. `4.0`) to require longer closure

### Too few / no alarms when sleepy
- Raise `ear_threshold` (e.g. `0.26`)
- Lower `perclos_threshold` (e.g. `0.25`)

### Windows: `UnicodeEncodeError` in terminal
- Run from **Windows Terminal** or **PowerShell** instead of the old Command Prompt
- Or set: `set PYTHONIOENCODING=utf-8` before running

### MediaPipe model download fails
- Manually download from:  
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`  
  and place it in the same folder as `sleep_alarm2.py`

---

## 📁 Project Structure

```
sleep-alarm/
├── sleep_alarm2.py          # Main application (all-in-one)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # MIT License
├── .gitignore               # Excludes model + reports from git
└── face_landmarker.task     # Auto-downloaded on first run (not in git)
```

---

## 🧪 How the Detection Works

```
Webcam frame  →  MediaPipe Face Landmarker (478 landmarks)
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
     Eye landmarks      Mouth landmarks      Nose/cheek pts
          │                   │                    │
        EAR calc            MAR calc          Pitch + gaze
          │                   │                    │
   Eyes closed?          Yawning?           Nodding/away?
          │                   │                    │
          └───────────────────┴────────────────────┘
                              │
                    Drowsiness Score (0–100)
                    [EAR×0.30 + PERCLOS×0.30 +
                     Blink×0.15 + Nod×0.15 + Yawn×0.10]
                              │
                    Score > threshold?
                       YES → 🔔 Alarm
```

**EAR formula** (Soukupová & Čech, 2016):
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```
Where p1–p6 are the 6 eye landmark points. EAR ≈ 0.3 when open, ≈ 0.0 when closed.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) — Face Landmarker model & framework
- [OpenCV](https://opencv.org/) — Computer vision & rendering
- PERCLOS standard — *Wierwille & Ellsworth (1994), "Research on Drowsy Driving"*
- EAR method — *Soukupová & Čech (2016), "Real-Time Eye Blink Detection"*

---

## 📜 License

MIT © 2026 Manish — See [LICENSE](LICENSE) for full text.

> Free to use, modify, and distribute. Attribution appreciated but not required.
