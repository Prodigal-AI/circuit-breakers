import os
import openpifpaf
import cv2
import torch
import numpy as np
from openpifpaf import decoder, show, transforms

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor' if device == torch.device('cpu') else 'torch.cuda.FloatTensor')

# Load model
predictor = openpifpaf.Predictor(checkpoint='resnet50')

# Exercise folders to process
EXERCISE_FOLDERS = ['leg raises','plank', 'pull Up', 'push-up', 'squat']  
OUTPUT_FOLDER = 'keypoints_data'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Keypoint extraction function
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, _, _ = predictor.numpy_image(img_rgb)

        if predictions:
            keypoints = predictions[0].data[:, :2]  # [17 x 2]
        else:
            keypoints = np.zeros((17, 2))  # fallback for missed frame

        keypoints_sequence.append(keypoints)

    cap.release()
    return np.array(keypoints_sequence)  # [frames x 17 x 2]

# Main processing loop
for folder in EXERCISE_FOLDERS:
    input_path = os.path.join('dataset', folder)
    output_path = os.path.join(OUTPUT_FOLDER, folder)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if filename.endswith('.mp4'):
            try:
                npy_filename = filename.replace('.mp4', '.npy')
                output_file = os.path.join(output_path, npy_filename)

                if os.path.exists(output_file):
                    print(f"Skipping {filename} in {folder} (already processed)")
                    continue

                full_path = os.path.join(input_path, filename)
                keypoints = extract_keypoints_from_video(full_path)
                np.save(output_file, keypoints)
                print(f"Processed {filename} in {folder}")

            except Exception as e:
                print(f"Error processing {filename} in {folder}: {e}")
