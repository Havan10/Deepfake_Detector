import cv2
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_VIDEO_DIR = os.path.join(BASE_DIR, "dataset", "real_videos")
FAKE_VIDEO_DIR = os.path.join(BASE_DIR, "dataset", "fake_videos")

REAL_FRAME_DIR = os.path.join(BASE_DIR, "dataset", "real_frames")
FAKE_FRAME_DIR = os.path.join(BASE_DIR, "dataset", "fake_frames")

os.makedirs(REAL_FRAME_DIR, exist_ok=True)
os.makedirs(FAKE_FRAME_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_frames(video_dir, frame_dir, max_frames=10):
    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)

        if not video.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            continue

        frame_indices = sorted(random.sample(range(total_frames), min(max_frames, total_frames)))

        current = 0
        saved = 0
        video_name = os.path.splitext(video)[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current in frame_indices:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (224, 224))

                    name = f"{video_name}_{saved}.jpg"
                    cv2.imwrite(os.path.join(frame_dir, name), face)

                    saved += 1
                    break

            current += 1

        cap.release()
        print(f"{video} → {saved} frames")


print("Extracting REAL...")
extract_frames(REAL_VIDEO_DIR, REAL_FRAME_DIR)

print("Extracting FAKE...")
extract_frames(FAKE_VIDEO_DIR, FAKE_FRAME_DIR)

print("Done!")