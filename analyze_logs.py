import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# -----------------------------
# 1. Input CSV
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python analyze_logs.py <csv_path>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

scenario = df["scenario"].iloc[0]

# Create folder if missing
os.makedirs("plots", exist_ok=True)

# -----------------------------
# 2. Face presence (0/1)
# -----------------------------
df["face_present"] = df["face_names"].apply(lambda x: 0 if x == "" else 1)

plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["face_present"], linewidth=2)
plt.title(f"Face Presence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Presence (1=yes, 0=no)")
plt.grid()
plt.savefig(f"plots/{scenario}_face_presence.png")
plt.close()


# -----------------------------
# 3. Face confidence curve
# -----------------------------
def extract_highest_conf(x):
    if not isinstance(x, str) or x.strip() == "":
        return 0.0
    vals = [float(v) for v in x.split(";") if v.strip() != ""]
    return max(vals) if vals else 0.0

df["best_face_conf"] = df["face_confidences"].apply(extract_highest_conf)

plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["best_face_conf"], linewidth=2)
plt.title(f"Highest Face Confidence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid()
plt.savefig(f"plots/{scenario}_face_confidence.png")
plt.close()


# -----------------------------
# 4. Object confidence timeline
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["object_confidence"], color="orange", linewidth=2)
plt.title(f"Object Confidence Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")
plt.grid()
plt.savefig(f"plots/{scenario}_object_confidence.png")
plt.close()


# -----------------------------
# 5. Detection quality timeline
# -----------------------------
df["quality_numeric"] = df["detection_quality"].apply(
    lambda x: 1 if x == "ok" else 0
)

plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["quality_numeric"], color="purple", linewidth=2)
plt.title(f"Detection Quality Over Time ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Quality (1 = OK, 0 = No Detection)")
plt.grid()
plt.savefig(f"plots/{scenario}_detection_quality.png")
plt.close()


# -----------------------------
# 6. Combined timeline plot
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["best_face_conf"], label="Best Face Confidence", linewidth=2)
plt.plot(df["timestamp"], df["object_confidence"], label="Object Confidence", linewidth=2)
plt.plot(df["timestamp"], df["face_present"] * df["best_face_conf"], 
         label="Weighted Face Presence", linewidth=2)

plt.title(f"Combined Detection Timeline ({scenario})")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.savefig(f"plots/{scenario}_combined_timeline.png")
plt.close()

print(f"[INFO] All graphs saved in /plots for scenario: {scenario}")
