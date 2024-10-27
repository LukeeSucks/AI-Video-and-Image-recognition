import cv2
import os

input_path = 'DogScene.mp4'
output_dir = 'test_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:06d}.png')
    cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if frame_count >= 10:  # Limit to 10 frames for testing
        break

cap.release()
print(f"Saved {frame_count} frames.")
