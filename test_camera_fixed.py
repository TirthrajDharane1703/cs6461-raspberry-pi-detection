from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (640, 480), "format": "YUV420"},
    display="main"
)
picam2.configure(config)

# FULL AUTO — let camera choose proper exposure and gain
picam2.set_controls({
    "AeEnable": True,    # auto exposure
    "AwbEnable": True,   # auto white balance
})

picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Test Auto", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
