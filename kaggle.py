import os
import cv2
import pandas as pd
import media_pipe

def collate_images(event_type):
    data_path = ""
    key_name = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if event_type == "train":
        data_path = os.path.join(script_dir, 'Train_Alphabet')
        key_name = 'trainKey'
    else:
        data_path = os.path.join(script_dir, 'Test_Alphabet')
        key_name = 'testKey'

    # 1. Setup MediaPipe HandLandmarkerManager
    landmarker = media_pipe.HandLandmarkerManager()
    
    if landmarker.landmarker is None:
        print("Error: HandLandmarker not initialized. Check model path.")
        return []

    # 2. RELATIVE PATH: This looks for the 'archive/letters' folder 
    # starting from wherever this script is saved.

    output_data = []
    output_key = []

    if not os.path.exists(data_path):
        print(f"Error: Could not find path '{data_path}'. Check your folder structure!")
    else:
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if not os.path.isdir(label_path): 
                continue
            if label == 'Blank':
                continue
            print(f"Processing letter: {label}")
            #count = len(os.listdir(label_path)) # If we want less data
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Extract landmarks
                result = landmarker.extract_landmarks(img)
                if result and result.hand_landmarks:
                    # Save landmark data
                    for hand_landmarks in result.hand_landmarks:
                        numeric_label = ord(label.upper()) - ord('A')  # Result: 0
                        row_data = {}
                        row_key = {'label': numeric_label}

                        # Handle both object with .landmark attribute and direct lists
                        landmarks_to_iterate = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
                        for idx, landmark in enumerate(landmarks_to_iterate):
                            row_data[f'x_{idx}'] = landmark.x
                            row_data[f'y_{idx}'] = landmark.y
                            row_data[f'z_{idx}'] = landmark.z
                        output_data.append(row_data)
                        output_key.append(row_key)

        # 3. Save the results
        if output_data:
            df = pd.DataFrame(output_data)
            df.to_csv(f"{event_type}.csv", index=False)
            print(f"Done! Saved {len(output_data)} hand landmark samples to {event_type}.csv'")
        else:
            print("No hand landmarks detected in any images.")
        if output_key:
            df = pd.DataFrame(output_key)
            df.to_csv(f"{key_name}.csv", index=False)
            print(f"Done! Saved {len(output_data)} hand landmark samples to {key_name}.csv'")
        else:
            print("No hand landmarks detected in any images.")
        
        landmarker.close()
        return output_data

