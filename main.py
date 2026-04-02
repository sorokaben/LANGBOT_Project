import opencv
import media_pipe
import time
import cv2 as cv
import kaggle
import os

def main():
    # Check if landmarks CSV exists, if not, batch process images
    if not os.path.exists('hand_landmarks.csv'):
        print("hand_landmarks.csv not found. Batch processing images...")
        kaggle.collate_images()
    else:
        print("hand_landmarks.csv already exists. Skipping batch processing.")
    
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

