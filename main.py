import cv2
import numpy as np
import os
import json
import random
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import subprocess

# Function to convert hex color code to BGR tuple
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color code to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return rgb[2], rgb[1], rgb[0]  # Convert RGB to BGR

# Function to create a gradient between two colors
def create_gradient(color1: Tuple[int, int, int], color2: Tuple[int, int, int], length: int) -> List[Tuple[int, int, int]]:
    gradient = []
    for i in range(length):
        ratio = i / length
        b = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        r = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        gradient.append((b, g, r))
    return gradient

# Load settings from JSON configuration file
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

# Function to check if the file is an image
def is_image(file_path: str) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return os.path.splitext(file_path)[1].lower() in image_extensions

# Function to generate output file path
def get_output_path(input_path: str, suffix: str = '_AR') -> str:
    directory, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}{suffix}{ext}"
    return os.path.join(directory, output_filename)


# TrackedObject class for object tracking
class TrackedObject:
    def __init__(self, object_id: int, class_id: int, centroid: Tuple[int, int],
                 bbox: List[int], annotate: bool, confidence: float):
        self.object_id = object_id
        self.class_id = class_id
        self.centroid = np.array(centroid, dtype=np.float32)
        self.bbox = bbox
        self.annotate = annotate
        self.frames_since_seen = 0
        self.confidence = confidence

        # Kalman filter initialization for smooth tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman.statePre = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)

    def predict(self):
        # Predict the next position using the Kalman filter
        predicted = self.kalman.predict()
        return int(predicted[0]), int(predicted[1])

    def update(self, measured_centroid: Tuple[int, int]):
        # Correct the Kalman filter with the new measurement
        self.kalman.correct(np.array(measured_centroid, dtype=np.float32))
        self.centroid = measured_centroid


# ObjectTracker class with Kalman filter integrated
class ObjectTracker:
    def __init__(self, max_distance: float = 50, max_frames_to_skip: int = 5,
                 classes: List[str] = None, classes_customization: Dict[str, Any] = None,
                 default_settings: Dict[str, Any] = None):
        self.next_object_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip
        self.classes = classes
        self.classes_customization = classes_customization
        self.default_settings = default_settings

    def update(self, detections: List[Tuple[int, int]], class_ids: List[int],
               confidences: List[float], boxes: List[List[int]]):
        updated_tracked_objects = {}
        detection_centroids = detections
        detection_class_ids = class_ids
        detection_boxes = boxes
        detection_confidences = confidences

        object_ids = list(self.tracked_objects.keys())
        object_centroids = [obj.centroid for obj in self.tracked_objects.values()]

        if not self.tracked_objects:
            for i in range(len(detection_centroids)):
                class_id = detection_class_ids[i]
                centroid = detection_centroids[i]
                bbox = detection_boxes[i]
                confidence = detection_confidences[i]
                annotate = self.decide_annotation(class_id)
                tracked_obj = TrackedObject(self.next_object_id, class_id, centroid,
                                            bbox, annotate, confidence)
                updated_tracked_objects[self.next_object_id] = tracked_obj
                self.next_object_id += 1
        else:
            if detection_centroids:
                D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(detection_centroids), axis=2)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                assigned_rows = set()
                assigned_cols = set()

                for (row, col) in zip(rows, cols):
                    if D[row, col] > self.max_distance:
                        continue
                    object_id = object_ids[row]
                    assigned_rows.add(row)
                    assigned_cols.add(col)
                    obj = self.tracked_objects[object_id]

                    # Update object with Kalman filter for smooth movement
                    obj.update(detection_centroids[col])
                    obj.bbox = detection_boxes[col]
                    obj.frames_since_seen = 0
                    obj.confidence = detection_confidences[col]
                    updated_tracked_objects[object_id] = obj

                unassigned_rows = set(range(len(object_ids))) - assigned_rows
                for row in unassigned_rows:
                    object_id = object_ids[row]
                    obj = self.tracked_objects[object_id]
                    obj.frames_since_seen += 1
                    if obj.frames_since_seen <= self.max_frames_to_skip:
                        updated_tracked_objects[object_id] = obj

                unassigned_cols = set(range(len(detection_centroids))) - assigned_cols
                for col in unassigned_cols:
                    class_id = detection_class_ids[col]
                    centroid = detection_centroids[col]
                    bbox = detection_boxes[col]
                    confidence = detection_confidences[col]
                    annotate = self.decide_annotation(class_id)
                    tracked_obj = TrackedObject(self.next_object_id, class_id, centroid,
                                                bbox, annotate, confidence)
                    updated_tracked_objects[self.next_object_id] = tracked_obj
                    self.next_object_id += 1
            else:
                for obj in self.tracked_objects.values():
                    obj.frames_since_seen += 1
                    if obj.frames_since_seen <= self.max_frames_to_skip:
                        updated_tracked_objects[obj.object_id] = obj

        self.tracked_objects = updated_tracked_objects

    def decide_annotation(self, class_id: int) -> bool:
        class_name = self.classes[class_id]
        class_settings = self.classes_customization.get(class_name, self.default_settings)
        detection_percentage = class_settings.get('detection_percentage', 100)
        detection_percentage = min(max(detection_percentage, 0), 100)
        if detection_percentage == 100:
            return True
        elif detection_percentage == 0:
            return False
        else:
            return random.uniform(0, 100) < detection_percentage

    def get_tracked_objects(self) -> List[TrackedObject]:
        return list(self.tracked_objects.values())

# Function to add glow effect
def add_glow_effect(image, mask, ksize=(0, 0), sigma=5):
    glow = cv2.GaussianBlur(mask, ksize, sigma)
    return cv2.addWeighted(image, 1.0, glow, 0.5, 0)

# Function to draw futuristic bounding boxes with gradient and animation
def draw_futuristic_box(image, x, y, w, h, base_color, thickness, frame_count):
    # Create corner lines with gradient color
    corner_length = int(min(w, h) * 0.2)
    overlay = image.copy()
    # Calculate gradient colors
    gradient_colors = create_gradient(base_color, (255, 255, 255), corner_length)
    # Animate the gradient
    shift = frame_count % corner_length
    # Top-left corner
    for i in range(corner_length):
        color = gradient_colors[(i + shift) % corner_length]
        cv2.line(overlay, (x + i, y), (x + i + 1, y), color, thickness)
        cv2.line(overlay, (x, y + i), (x, y + i + 1), color, thickness)
    # Top-right corner
    for i in range(corner_length):
        color = gradient_colors[(i + shift) % corner_length]
        cv2.line(overlay, (x + w - i, y), (x + w - i - 1, y), color, thickness)
        cv2.line(overlay, (x + w, y + i), (x + w, y + i + 1), color, thickness)
    # Bottom-left corner
    for i in range(corner_length):
        color = gradient_colors[(i + shift) % corner_length]
        cv2.line(overlay, (x + i, y + h), (x + i + 1, y + h), color, thickness)
        cv2.line(overlay, (x, y + h - i), (x, y + h - i - 1), color, thickness)
    # Bottom-right corner
    for i in range(corner_length):
        color = gradient_colors[(i + shift) % corner_length]
        cv2.line(overlay, (x + w - i, y + h), (x + w - i - 1, y + h), color, thickness)
        cv2.line(overlay, (x + w, y + h - i), (x + w, y + h - i - 1), color, thickness)
    # Add glow effect
    mask = np.zeros_like(image)
    cv2.rectangle(mask, (x, y), (x + w, y + h), base_color, thickness)
    image = add_glow_effect(overlay, mask)
    return image

# Define the processing function
def process_frame(frame: np.ndarray, tracker: ObjectTracker, net: cv2.dnn_Net,
                  classes: List[str], config: Dict[str, Any], random_colors: np.ndarray,
                  class_ids_to_detect: List[int], frame_count: int) -> np.ndarray:
    (H, W) = frame.shape[:2]

    # Create a copy of the frame for processing
    frame_processed = frame.copy()

    # Calculate scaling factor based on the shorter side of the frame
    reference_dimension = 720  # Reference resolution
    scaling_factor = min(W, H) / reference_dimension
    scaling_factor = max(scaling_factor, 1.0)  # Ensure the scaling factor is at least 1.0

    # Determine the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, class IDs, and centers
    boxes, confidences, class_ids, centers = [], [], [], []

    # Load settings
    detection_settings = config['detection_settings']
    CONFIDENCE_THRESHOLD = detection_settings['confidence_threshold']
    NMS_THRESHOLD = detection_settings['nms_threshold']

    visualization_settings = config['visualization_settings']
    USE_RANDOM_COLORS = visualization_settings['use_random_colors']
    USE_CUSTOM_COLORS = visualization_settings['use_custom_colors']
    DRAW_CONNECTING_LINES = visualization_settings['connecting_lines']['draw']
    CONNECTING_LINES_PERCENTAGE = visualization_settings['connecting_lines'].get('connecting_lines_percentage', 100)
    classes_customization = visualization_settings.get('classes_customization', {})
    default_settings = visualization_settings.get('default_visualization', {})

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > CONFIDENCE_THRESHOLD and class_id in class_ids_to_detect:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width_box, height_box) = box.astype('int')

                x = int(centerX - (width_box / 2))
                y = int(centerY - (height_box / 2))

                boxes.append([x, y, int(width_box), int(height_box)])
                confidences.append(confidence)
                class_ids.append(class_id)
                centers.append((centerX, centerY))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        idxs = idxs.flatten()
        boxes = [boxes[i] for i in idxs]
        confidences = [confidences[i] for i in idxs]
        class_ids = [class_ids[i] for i in idxs]
        centers = [centers[i] for i in idxs]

    tracker.update(centers, class_ids, confidences, boxes)
    tracked_objects = tracker.get_tracked_objects()

    overlay = frame_processed.copy()
    for obj in tracked_objects:
        if obj.annotate:
            x, y, w, h = obj.bbox
            class_id = obj.class_id
            class_name = classes[class_id]
            confidence = obj.confidence

            class_settings = classes_customization.get(class_name, default_settings)
            base_color = hex_to_bgr(class_settings.get('bounding_box_color', '#00ff00'))

            bbox_thickness = int(class_settings.get('bounding_box_thickness', 2) * scaling_factor)
            bbox_thickness = max(1, bbox_thickness)

            # Draw the futuristic bounding box with animation
            overlay = draw_futuristic_box(overlay, x, y, w, h, base_color, bbox_thickness, frame_count)

            text = f"{class_name.upper()} {confidence * 100:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            label_x, label_y = x, y - 10
            cv2.putText(overlay, text, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    alpha = 1.0  # Full opacity
    overlay_float = overlay.astype(np.float32)
    frame_float = frame_processed.astype(np.float32)
    blended_float = cv2.addWeighted(overlay_float, alpha, frame_float, 1 - alpha, 0)
    frame_processed = np.clip(blended_float, 0, 255).astype(np.uint8)


    return frame_processed

# Function to assemble video
def assemble_video_with_ffmpeg(frames_dir: str, output_video_path: str, fps: float, width: int, height: int, high_quality: bool = True):
    ffmpeg_command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-y",
        "-i", f"{frames_dir}/frame_%06d.png",
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
    ]

    if high_quality:
        ffmpeg_command.extend(["-crf", "16", "-b:v", "10M"])
    else:
        ffmpeg_command.extend(["-crf", "23"])

    ffmpeg_command.append(output_video_path)
    print(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
    subprocess.run(ffmpeg_command, check=True)

def main():
    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askopenfilename(title="Select an image or video file")
    if not input_path:
        print("No file selected. Exiting.")
        exit()

    config = load_config('config.json')
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    class_name_to_id = {name: idx for idx, name in enumerate(classes)}
    detection_settings = config['detection_settings']
    classes_to_detect = detection_settings['classes_to_detect']

    if classes_to_detect:
        class_ids_to_detect = [class_name_to_id[cls] for cls in classes_to_detect if cls in class_name_to_id]
    else:
        class_ids_to_detect = list(range(len(classes)))

    visualization_settings = config['visualization_settings']
    output_settings = config['output_settings']

    USE_RANDOM_COLORS = visualization_settings['use_random_colors']
    if USE_RANDOM_COLORS:
        np.random.seed(42)
        random_colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    else:
        random_colors = None

    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    random.seed(42)
    output_path = get_output_path(input_path)
    output_dir, output_filename = os.path.split(output_path)
    output_name, _ = os.path.splitext(output_filename)

    if is_image(input_path):
        print(f"Processing image: {input_path}")
        tracker = ObjectTracker(classes=classes, classes_customization=visualization_settings.get('classes_customization', {}), default_settings=visualization_settings.get('default_visualization', {}))
        image = cv2.imread(input_path)

        frame_count = 0
        print("Running forward pass to get detections...")
        annotated_image = process_frame(image, tracker, net, classes, config, random_colors, class_ids_to_detect, frame_count)
        print("Image processing complete.")

        if output_settings['display_image']:
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if output_settings['save_output']:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to {output_path}")

    else:
        print(f"Processing video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            exit()
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        tracker = ObjectTracker(classes=classes, classes_customization=visualization_settings.get('classes_customization', {}), default_settings=visualization_settings.get('default_visualization', {}))

        frame_count = 0
        interrupted = False

        frames_dir = os.path.join(output_dir, f"{output_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            with tqdm(total=total_frames, desc='Processing Video', unit='frame') as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    annotated_frame = process_frame(frame, tracker, net, classes, config, random_colors, class_ids_to_detect, frame_count)

                    if output_settings['save_output']:
                        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:06d}.png')
                        cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    if output_settings['display_video']:
                        cv2.imshow('Annotated Video', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            interrupted = True
                            break

                    pbar.update(1)
        except KeyboardInterrupt:
            interrupted = True
        except Exception as e:
            interrupted = True
            print(f"\nAn error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

            if output_settings['save_output'] and frame_count > 0:
                print("Assembling frames into video using FFmpeg...")
                output_video_path = os.path.join(output_dir, f"{output_name}_AR.mp4")
                assemble_video_with_ffmpeg(frames_dir, output_video_path, fps, width, height, high_quality=True)
                print(f"Annotated video saved to {output_video_path}")

if __name__ == "__main__":
    main()
