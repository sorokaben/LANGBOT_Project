from __future__ import annotations

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from mediapipe import Image
from mediapipe.tasks.python.vision import HandLandmarkerResult
import cv2 as cv

class HandLandmarkerManager:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')
        print(f"Model path: {model_path}")  # Debug print
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            self.landmarker = None
            return

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        def print_result(result, output_image, timestamp_ms: int):
            print('hand landmarker result: {}'.format(result))

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
        self.landmarker = HandLandmarker.create_from_options(options)

    def extract_landmarks(self, numpy_frame_from_opencv):
        """Extract raw landmarks from a frame without drawing."""
        if self.landmarker is None:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        result = self.landmarker.detect(mp_image)
        return result
    
    def detect(self, numpy_frame_from_opencv, frame_timestamp_ms):
        """Detect landmarks and draw them on the frame."""
        if self.landmarker is None:
            return numpy_frame_from_opencv
        image = numpy_frame_from_opencv.copy()
        result = self.extract_landmarks(image)
        if result and result.hand_landmarks:
            for landmark in result.hand_landmarks[0]:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv.circle(image, (x, y), 5, (255,255,255), -1)  
        return image

    def close(self):
        if self.landmarker:
            self.landmarker.close()




