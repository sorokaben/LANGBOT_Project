import opencv
import media_pipe
import time
import cv2 as cv
import kaggle
import os
import AImodel

prev_letters = "n"
"""
def displayText(aFrame, letterGuess):
    global prev_letters
    cv.putText(aFrame, f'Letter: {letterGuess}', (50, 50), 
        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.putText(aFrame, f'{prev_letters}', (50, 200), 
        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if len(prev_letters) > 15:
        prev_letters = prev_letters[1:]
    prev_letters += letterGuess
"""

def extractLandmarks(landmark_result):
    # No hands detected
    if not landmark_result or not landmark_result.hand_landmarks:
        return []

    landmarks = []

    hand = landmark_result.hand_landmarks[0]

    for mark in hand:
        landmarks.append(mark.x)
        landmarks.append(mark.y)

    return landmarks
        

def main():
    # Check if landmarks CSV exists, if not, batch process images
    if not os.path.exists('train.csv'):
        print("train.csv not found. Batch processing images...")
        kaggle.collate_images('train')
        kaggle.collate_images('test')

    else:
        print("train.csv already exists. Skipping batch processing.")
    
    landmarker = media_pipe.HandLandmarkerManager()

    testModel = AImodel.ASL_model()
        
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


            handCoordinates = extractLandmarks(landmarker.extract_landmarks(frame))

            if not handCoordinates:
                print("No hand detected")
            else:
                # displayText(annotated_frame, testModel.read_sign(handCoordinates))
                print(testModel.read_sign(handCoordinates))

            cv.imshow('Hand Landmarks', annotated_frame)

            time.sleep(0.0666666)  # ~30 FPS
    finally:
        landmarker.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()


