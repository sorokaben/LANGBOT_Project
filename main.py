import opencv
import media_pipe
import time

def main():
    landmarker = media_pipe.HandLandmarkerManager()
    try:
        while True:
            frame = opencv.capture_video()
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect(frame, timestamp_ms)
            time.sleep(0.033)  # ~30 FPS
    finally:
        landmarker.close()

if __name__ == "__main__":
    main()

