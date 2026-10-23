import os
import cv2
import face_recognition
import pickle

DATA_DIR = "data/faces"
ENCODINGS_PATH = "data/encodings.pkl"

image_paths = []
names = []

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)

    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        if img_name.lower().endswith((".jpg", ".png")):
            image_paths.append(os.path.join(person_dir, img_name))
            names.append(person)

print(f"[INFO] Found {len(image_paths)} face images.")

encodings = []

for img_path, name in zip(image_paths, names):
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    if len(boxes) == 0:
        print(f"[WARN] No face found in {img_path}")
        continue

    enc = face_recognition.face_encodings(rgb, boxes)[0]
    encodings.append(enc)

print("[INFO] Saving encodings to file...")
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print("[INFO] Done.")
