from __future__ import annotations

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from mediapipe import Image
from mediapipe.tasks.python.vision import HandLandmarkerResult

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
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, numpy_frame_from_opencv, frame_timestamp_ms):
        if self.landmarker is None:
            return
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        self.landmarker.detect_async(mp_image, frame_timestamp_ms)

    def close(self):
        if self.landmarker:
            self.landmarker.close()

