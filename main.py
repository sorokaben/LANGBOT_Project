import opencv
import media_pipe
import time
import cv2 as cv

def main():
    landmarker = media_pipe.HandLandmarkerManager()
    try:
        while True:
            frame = opencv.capture_frame()
            if frame is None:
                break
            if frame is -1:
                print("Q detected. Exiting")
                exit(1)
            timestamp_ms = int(time.time() * 1000)
            annotated_frame = landmarker.detect(frame, timestamp_ms)
            cv.imshow('Hand Landmarks', annotated_frame)

            time.sleep(0.033)  # ~30 FPS
    finally:
        landmarker.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()

