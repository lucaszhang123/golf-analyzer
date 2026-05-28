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
├── main.py                    ← entry point, run this
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

### Basic analysis (face view)
```bash
python3 main.py swing.mov
```

### Basic analysis (back view)
```bash
python3 main.py swing.mov --back
```

### With stick figure video
```bash
python3 main.py swing.mov --stick
python3 main.py swing.mov --back --stick
```

### Skip the annotated video (faster, just metrics + faults)
```bash
python3 main.py swing.mov --no-video --no-feedback
```

### Specify club
```bash
python3 main.py swing.mov --club driver
python3 main.py swing.mov --club 7i
python3 main.py swing.mov --club pw
```
Valid clubs: `driver`, `3w`, `5w`, `7w`, `hybrid`, `2i` `3i` `4i` `5i` `6i` `7i` `8i` `9i`, `pw` `gw` `sw` `lw`

### Left-handed golfer
```bash
python3 main.py swing.mov --hand left
```

### With AI coaching feedback (requires Anthropic API key)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 main.py swing.mov --club 7i

# Or pass key directly
python3 main.py swing.mov --club 7i --api-key sk-ant-...
```

### Organize multiple swings into named sessions
```bash
# Groups all outputs for one session together
python3 main.py swing_face.mov --name "session1" --club 7i --stick
python3 main.py swing_back.mov --name "session1" --back --stick
```

---

## Output Structure

Without `--name`:
```
output/
├── face/
│   ├── snapshots/          ← annotated PNG per frame
│   ├── stick_frames/       ← stick figure PNG per frame
│   ├── swing_analyzed.mp4  ← annotated video
│   └── swing_stick.mp4     ← stick figure video (8fps slow motion)
└── back/
    ├── snapshots/
    ├── stick_frames/
    ├── swing_back.mp4
    └── swing_back_stick.mp4
```

With `--name "session1"`:
```
output/
└── session1/
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

```
python3 main.py [video] [options]

Options:
  --back              Analyze as back view (default: face view)
  --club CLUB         Club used: driver, 7i, pw, etc. (default: 7i)
  --hand {right,left} Golfer's dominant hand (default: right)
  --name NAME         Session name for organized output folders
  --stick             Generate side-by-side stick figure video
  --stick-only        Stick figure on black only (no original alongside)
  --no-video          Skip annotated video output (faster)
  --no-feedback       Skip AI coaching feedback
  --api-key KEY       Anthropic API key for AI feedback
  --max-faults N      Max faults to get AI feedback on (default: 3)
  --snapshots N       Limit to N evenly-spaced frames (default: every frame)
  --output PATH       Custom output video path
```

---

## Tips

- **Slow-mo videos work** — the analyzer processes every frame. High frame rate = more data through the fast downswing.
- **Face view** is filmed perpendicular to the target line (camera facing golfer directly).
- **Back view** is filmed from behind, looking toward the target.
- **Stick figure videos** play at 8fps (slow motion) so you can see each frame clearly.
- **Confidence score** in the overlay shows how reliably MediaPipe detected the skeleton — below 50% means the frame is unreliable.
- **Session names** keep multiple swings organized — use `--name "driver_round1"` etc.