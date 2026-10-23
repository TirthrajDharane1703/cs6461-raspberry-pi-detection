from picamera2 import Picamera2
import cv2
import os

DATA_DIR = "data/faces"

def capture_person(name, num_images=20):
    os.makedirs(f"{DATA_DIR}/{name}", exist_ok=True)

    picam2 = Picamera2()

    # Correct ISP configuration
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        lores={"size": (640, 480), "format": "YUV420"},
        display="main"
    )
    picam2.configure(config)

    # Use full auto AE/AWB (works best now)
    picam2.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "Sharpness": 1.2,
        "Contrast": 1.1
    })

    picam2.start()
    print("[INFO] Press SPACE to capture image, Q to quit.")

    count = 0
    while count < num_images:
        frame = picam2.capture_array()

        disp = frame.copy()
        cv2.putText(disp, f"{name} {count+1}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Capture Faces", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            out_path = f"{DATA_DIR}/{name}/{name}_{count}.jpg"
            cv2.imwrite(out_path, frame)
            print("Saved", out_path)
            count += 1

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    print(f"[INFO] Finished capturing {count} images for {name}.")


if __name__ == "__main__":
    name = input("Enter person name (no spaces): ")
    capture_person(name)
