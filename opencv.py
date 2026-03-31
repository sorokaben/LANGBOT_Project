import numpy as np
import cv2 as cv

cap = None

def capture_video():
    global cap
    if cap is None:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return None
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        return None
    # Return the color frame (BGR), as MediaPipe needs color for hand detection
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        return
    return frame

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

