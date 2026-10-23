import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# ----------------------------
# Load CSV
# ----------------------------
path = sys.argv[1]
df = pd.read_csv(path)

scenario = df["scenario"][0]
timestamps = df["timestamp"]

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

# ----------------------------
# Face Presence Plot
# ----------------------------
df["face_present"] = df["face_names"].apply(lambda x: 0 if x == "" else 1)

plt.figure(figsize=(10,5))
plt.plot(timestamps, df["face_present"], label="Face Present")
plt.title(f"Face Presence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Presence (1=yes, 0=no)")
plt.grid(True)
plt.savefig(f"plots/{scenario}_face_presence.png")
plt.close()

# ----------------------------
# Face Confidence Plot
# ----------------------------
face_conf = []

for x in df["face_confidences"]:
    if x == "" or pd.isna(x):
        face_conf.append(0)
    else:
        # Take max confidence if multiple names (e.g., "0.88;0.65")
        vals = [float(v) for v in str(x).split(";")]
        face_conf.append(max(vals))

plt.figure(figsize=(10,5))
plt.plot(timestamps, face_conf, label="Face Confidence")
plt.title(f"Face Confidence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid(True)
plt.savefig(f"plots/{scenario}_face_confidence.png")
plt.close()

# ----------------------------
# Object Confidence Plot
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(timestamps, df["object_confidence"])
plt.title(f"Object Confidence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid(True)
plt.savefig(f"plots/{scenario}_object_confidence.png")
plt.close()

# ----------------------------
# Detection Quality Timeline
# ----------------------------
quality_map = {"ok": 1, "no_detection": 0}
df["quality_val"] = df["detection_quality"].map(quality_map)

plt.figure(figsize=(10,4))
plt.plot(timestamps, df["quality_val"])
plt.title(f"Detection Quality Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("1 = OK, 0 = No Detection")
plt.grid(True)
plt.savefig(f"plots/{scenario}_detection_quality.png")
plt.close()

# ----------------------------
# Names Timeline (Gantt Chart)
# ----------------------------

# Extract all unique names encountered
all_names = set()
for row in df["face_names"]:
    if row != "" and not pd.isna(row):
        for n in str(row).split(";"):
            all_names.add(n)

all_names = sorted(list(all_names))

presence_matrix = {name: [] for name in all_names}

for row in df["face_names"]:
    detected_now = set(str(row).split(";")) if row != "" else set()
    for name in all_names:
        presence_matrix[name].append(1 if name in detected_now else 0)

# Plot timeline
plt.figure(figsize=(12, 6))

for i, name in enumerate(all_names):
    y_vals = np.array(presence_matrix[name]) * (i+1)
    plt.plot(timestamps, y_vals, label=name, linewidth=6)

plt.yticks(range(1, len(all_names)+1), all_names)
plt.xlabel("Time (s)")
plt.title(f"Names Timeline ({scenario})")
plt.grid(True)
plt.savefig(f"plots/{scenario}_combined_timeline.png")
plt.close()

print(f"All plots saved in /plots for scenario: {scenario}")
