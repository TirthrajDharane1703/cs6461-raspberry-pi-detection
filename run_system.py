from picamera2 import Picamera2
import cv2
import face_recognition
import pickle
import time
import csv
import json
import sys
from aruco_utils import detect_aruco

# -------------------------
# LOAD SCENARIO CONFIG
# -------------------------
if len(sys.argv) != 2:
    print("Usage: python run_system.py '<json_string>'")
    sys.exit(1)

cfg = json.loads(sys.argv[1])
SCENARIO_ID = cfg["scenario_id"]
LOG_PATH = cfg["log_path"]

print(f"[INFO] Running scenario: {SCENARIO_ID}")
print("[INFO] Press Q to stop recording.")

# -------------------------
# LOAD FACE ENCODINGS
# -------------------------
ENCODINGS_PATH = "data/encodings.pkl"
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# -------------------------
# INITIALIZE CAMERA
# -------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (640, 480), "format": "YUV420"},
    display="main"
)
picam2.configure(config)
picam2.set_controls({"AeEnable": True, "AwbEnable": True})
picam2.start()
time.sleep(1)

# -------------------------
# OPEN LOG FILE
# -------------------------
with open(LOG_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "timestamp","scenario","face_names","face_confidences",
        "object_detected","object_confidence","detection_quality"
    ])
    writer.writeheader()

    start_time = time.time()

    # ------------------------------------
    # MAIN LOOP — RUN UNTIL USER PRESSES Q
    # ------------------------------------
    while True:

        frame = picam2.capture_array()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------
        # FACE DETECTION
        # -------------------------
        face_locations = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        names_out = []
        confs_out = []

        for enc in face_encs:
            dists = face_recognition.face_distance(known_encodings, enc)
            idx = dists.argmin()
            conf = max(0.0, 1.0 - dists[idx])

            name = known_names[idx]
            names_out.append(name)
            confs_out.append(round(float(conf), 3))

        # Draw boxes on preview
        for (top, right, bottom, left), name, conf in zip(face_locations, names_out, confs_out):
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, f"{name} {conf}", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # -------------------------
        # ARUCO OBJECT DETECTION
        # -------------------------
        obj_found, obj_box, obj_conf = detect_aruco(frame)

        if obj_found:
            (x, y, w, h) = obj_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, f"OBJ {obj_conf}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        # -------------------------
        # WRITE LOG ENTRY
        # -------------------------
        elapsed = time.time() - start_time

        writer.writerow({
            "timestamp": round(elapsed, 3),
            "scenario": SCENARIO_ID,
            "face_names": ";".join(names_out),
            "face_confidences": ";".join(map(str, confs_out)),
            "object_detected": "ArucoObj" if obj_found else "",
            "object_confidence": round(float(obj_conf), 3) if obj_found else 0.0,
            "detection_quality": "ok" if names_out or obj_found else "no_detection"
        })

        # -------------------------
        # SHOW PREVIEW WINDOW
        # -------------------------
        cv2.imshow("CS6461 System Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Q pressed → stopping.")
            break

picam2.stop()
cv2.destroyAllWindows()
