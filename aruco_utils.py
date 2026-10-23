import cv2
import numpy as np

def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    if ids is not None and len(ids) > 0:
        # Only use the first detected marker
        c = corners[0].reshape(4, 2)

        # Compute bounding box
        x_min = int(c[:, 0].min())
        y_min = int(c[:, 1].min())
        x_max = int(c[:, 0].max())
        y_max = int(c[:, 1].max())

        w = x_max - x_min
        h = y_max - y_min

        # Fake confidence
        conf = 0.99  

        return True, (x_min, y_min, w, h), conf

    return False, (0, 0, 0, 0), 0.0
