# Golf Swing Analyzer

Analyzes golf swing videos using MediaPipe pose detection, biomechanical fault rules, and Claude AI coaching feedback.

---

## Setup (one time)

**1. Install Python dependencies**
```bash
pip3 install mediapipe opencv-python numpy
```

**2. On first run**, the pose model (~7MB) downloads automatically to `~/.golf_analyzer/`. No manual steps needed.

---

## File Structure

```
golf_analyzer/
├── main.py                    ← step 1: capture pose, metrics, stick videos
├── faults.py                  ← step 2: run fault detection on a session
├── README.md
├── your_video.mov             ← put your videos here
└── core/
    ├── analyzer.py            ← face view pipeline
    ├── analyzer_back.py       ← back view pipeline
    ├── metrics.py             ← face view measurements
    ├── metrics_back.py        ← back view measurements
    ├── fault_rules.py         ← face view fault detection
    ├── fault_rules_back.py    ← back view fault detection
    ├── fault_detector.py      ← fault orchestrator
    ├── annotator.py           ← draws skeleton on frames
    ├── stick_figure.py        ← black background stick figure
    ├── landmarks.py           ← MediaPipe joint constants
    ├── feedback.py            ← Claude AI coaching feedback
    └── club_detector.py       ← club shaft detection (experimental)
```

---

## How to Run

The pipeline is split into **two steps**:

1. **`main.py`** — capture step. Extracts pose, snapshots, stick figure videos,
   and writes per-frame metrics to `metrics.json`. **No fault detection runs
   here.**
2. **`faults.py`** — analysis step. Reads the captured session and runs fault
   detection (and optionally LLM coaching feedback) on whichever views are
   present.

This means you can capture once and re-run fault analysis as many times as you
want (e.g. with a different club, with/without AI feedback) without
re-processing video.

### Step 1 — Capture (recommended: both views at once)
```bash
python3 main.py --face swing.mov --back swing_back.mov --club 7i
```

This creates the next free `output/sessionN/` and writes:
- `output/sessionN/face/` — face-view snapshots, stick frames, metrics.json, annotated MP4
- `output/sessionN/back/` — back-view snapshots, stick frames, metrics.json, annotated MP4
- `output/sessionN/session.json` — manifest used by `faults.py`

The console output only shows pose extraction progress — no fault report.

### Step 2 — Fault detection
```bash
python3 faults.py output/sessionN/
```

Runs the syndrome-based fault detector on both views and prints reports.

#### With AI coaching feedback
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 faults.py output/sessionN/

# Or pass the key directly:
python3 faults.py output/sessionN/ --api-key sk-ant-...
```

#### Override the captured club
```bash
python3 faults.py output/sessionN/ --club driver
```

#### Save the fault reports to JSON
```bash
python3 faults.py output/sessionN/ --save-json
# Writes faults_face.json + faults_back.json into the session folder.
```

#### Only one view
```bash
python3 faults.py output/sessionN/ --only face
python3 faults.py output/sessionN/ --only back
```

### Other capture options

Just one view:
```bash
python3 main.py --face swing.mov
python3 main.py --back swing_back.mov
```

Skip the annotated video (faster — stick + metrics only):
```bash
python3 main.py --face swing.mov --back swing_back.mov --no-video
```

Skip stick figure rendering entirely:
```bash
python3 main.py --face swing.mov --back swing_back.mov --no-stick
```

Stick-only (no original side-by-side):
```bash
python3 main.py --face swing.mov --back swing_back.mov --stick-only
```

Custom session name (otherwise auto sessionN):
```bash
python3 main.py --face swing.mov --back swing_back.mov --name "driver_round1"
```

Left-handed golfer:
```bash
python3 main.py --face swing.mov --hand left
```

Valid clubs: `driver`, `3w`, `5w`, `7w`, `hybrid`, `2i` `3i` `4i` `5i` `6i` `7i` `8i` `9i`, `pw` `gw` `sw` `lw`

---

## Output Structure

Every run goes into a single session folder so face and back never get separated.

After `main.py` (auto-named, default):
```
output/
└── session1/                  ← next free sessionN
    ├── session.json            ← manifest read by faults.py
    ├── face/
    │   ├── snapshots/          ← annotated PNG per frame
    │   ├── stick_frames/       ← stick figure PNG per frame
    │   ├── metrics.json        ← per-frame metrics (input to faults.py)
    │   ├── swing_analyzed.mp4  ← annotated video
    │   └── swing_stick.mp4     ← stick figure video (8fps slow motion)
    └── back/
        ├── snapshots/
        ├── stick_frames/
        ├── metrics.json
        ├── swing_back.mp4
        └── swing_back_stick.mp4
```

After `faults.py --save-json`:
```
output/session1/
├── faults_face.json
└── faults_back.json
```

With `--name "driver_round1"`:
```
output/
└── driver_round1/
    ├── session.json
    ├── face/
    └── back/
```

---

## What Each View Analyzes

### Face view (camera facing golfer)
| Metric | What it measures |
|---|---|
| Shoulder turn | Rotation from address (elite: ~95°) |
| Hip turn | Hip rotation at top (elite: ~45°) |
| X-Factor | Shoulder minus hip separation (elite: ~56°) |
| Spine tilt | Lateral lean during swing |
| Weight shift | Lead/trail foot loading (-1 trail, +1 lead) |
| Lead elbow | Arm structure through impact |
| Wrist hinge | Wrist cock at top of backswing |
| Head movement | Lateral stability from address |
| Hip sway | Lateral hip slide vs rotation |
| Knee flex | Lead and trail knee bend |

### Back view (camera behind golfer)
| Metric | What it measures |
|---|---|
| Spine angle | Forward bend from vertical (ideal: 30-45° at address) |
| Spine maintenance | Angle change from address to impact (<8° ideal) |
| Lead arm plane | Above/on/below shoulder plane at top |
| Shoulder plane | Steepening/shallowing in downswing |
| Trail knee flex | Knee maintains flex in backswing |
| Lead knee flex | Knee maintains flex in backswing |
| Hip slide | Lateral movement toward target in downswing |
| Head movement | Forward/backward stability |

---

## All Command Options

### `main.py` — capture
```
python3 main.py [--face FACE_VIDEO] [--back BACK_VIDEO] [options]

Inputs (pass one or both):
  --face, -f PATH     Face-on swing video
  --back, -b PATH     Back / down-the-line swing video

Options:
  --name NAME         Session folder name (default: auto sessionN)
  --club CLUB         Club used (recorded into session.json) (default: 7i)
  --hand {right,left} Golfer's dominant hand (default: right)
  --snapshots N       Limit to N evenly-spaced frames (default: every frame)
  --no-video          Skip annotated MP4 output (faster)
  --stick-only        Stick figure on black only (no original alongside)
  --no-stick          Skip stick figure rendering entirely
```

### `faults.py` — fault detection + AI feedback
```
python3 faults.py <session-folder> [options]

Options:
  --only {face,back,both}  Limit analysis to one view (default: both)
  --club CLUB              Override club from session.json
  --no-feedback            Skip LLM coaching feedback
  --api-key KEY            Anthropic API key (else $ANTHROPIC_API_KEY)
  --max-faults N           Max faults to send to LLM (default: 3)
  --save-json              Write faults_face.json / faults_back.json
```

---

## Tips

- **Slow-mo videos work** — the analyzer processes every frame. High frame rate = more data through the fast downswing.
- **Face view** is filmed perpendicular to the target line (camera facing golfer directly).
- **Back view** is filmed from behind, looking toward the target.
- **Stick figure videos** play at 8fps (slow motion) so you can see each frame clearly.
- **Confidence score** in the overlay shows how reliably MediaPipe detected the skeleton — below 50% means the frame is unreliable.
- **Session names** keep multiple swings organized — use `--name "driver_round1"` etc.