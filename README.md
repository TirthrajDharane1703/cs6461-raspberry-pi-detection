# Real-Time Multimodal Person and Object Detection System

CS6461 Computer Vision Systems — module project, Raspberry Pi 4.

A real-time detection system built on a Raspberry Pi 4 that combines face recognition
(`face_recognition`) with ArUco marker-based object detection (OpenCV), tuning the
camera's own ISP pipeline (auto-exposure, auto white balance, contrast/sharpness) and
testing it systematically across three distinct lighting scenarios — indoor tungsten,
bright/outdoor daylight, and evening low-light — to analyse where and why detection
confidence degrades under real sensor and lighting conditions.

## How it works

1. `run_system.py` captures frames from the Pi camera via the Picamera2 API, applies
   AWB/exposure/contrast ISP configuration, runs face recognition against a set of
   pre-enrolled face encodings, and detects a printed ArUco marker (ID 0) as the
   required object.
2. Each frame's detections (face identity + confidence, object presence + confidence,
   overall detection quality) are logged to a per-run CSV.
3. `analyze_logs.py` / `analyze_results.py` turn those CSVs into timeline and
   confidence plots (`plots/`) used to compare detection behaviour across scenarios.

## Scripts

| Script | Purpose |
|---|---|
| `run_system.py` | Main real-time detection loop (face recognition + ArUco), logs to CSV |
| `run_scenario.py` | Runs `run_system.py` against a given scenario config |
| `register_faces.py` | Captures enrollment images for a new person via the Pi camera |
| `encode_faces.py` | Builds `data/encodings.pkl` from enrolled face images |
| `aruco_utils.py` | ArUco marker detection helper |
| `analyze_logs.py` / `analyze_results.py` | Generate plots from run logs |
| `test_awb.py`, `test_camera2.py`, `test_camera_fixed.py` | Camera/ISP tuning test scripts |

## Scenarios

Three lighting conditions were tested (`scenarios/*.json`), each producing a logged
run and a set of plots in `plots/`:
- Indoor tungsten light (strong yellow cast, AWB-compensation-dependent)
- Bright / outdoor daylight (strong directional light, silhouette effects)
- Evening low-light (mixed lighting, lowest detection confidence)

## Running it

```bash
pip install -r requirements.txt

# enroll a face
python register_faces.py
python encode_faces.py

# run a scenario
python run_scenario.py scenarios/indoor_tungsten.json

# generate plots from a run's log
python analyze_logs.py logs/run_indoor_tungsten.csv
```

## Note on data

Enrolled face images, the derived face-encoding file (`data/encodings.pkl`), and the
demo video are excluded from this repo since they contain other people's photos. The
code, scenario configs, run logs, and resulting plots are included.
