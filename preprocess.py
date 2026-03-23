import cv2
import os

def extract_frames(video_path, output_folder, max_frames=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, cv2.resize(frame, (224, 224)))
        count += 1
    cap.release()

# Set your dataset paths
base_path = "dataset/train"
output_base = "frames_data"

for category in ['Real', 'Fake']:
    folder_path = os.path.join(base_path, category)
    for video in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video)
        output_folder = os.path.join(output_base, category, video.split('.')[0])
        extract_frames(video_path, output_folder)
print("Preprocessing Complete!")