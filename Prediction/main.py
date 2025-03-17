import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Set paths
MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt"
TEST_IMAGES_DIR = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/Model_Fabric_Defect_Detection/detected_objects"
GROUND_TRUTH_DIR = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/defect-dataset/test/labels"

# Check if CUDA (GPU) is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load YOLO model on GPU
model = YOLO(MODEL_PATH).to(DEVICE)

# IoU Threshold for TP detection
IOU_THRESHOLD = 0.5

# Function to calculate IoU (Intersection over Union)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection
    xi1, yi1 = max(x1, x1_gt), max(y1, y1_gt)
    xi2, yi2 = min(x2, x2_gt), min(y2, y2_gt)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Function to read YOLO ground truth labels
def read_ground_truth(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append((x1, y1, x2, y2, class_id))

    return boxes

# Counters for evaluation metrics
tp, fp, fn = 0, 0, 0

# Process test images
for image_name in os.listdir(TEST_IMAGES_DIR):
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):  # Skip non-image files
        continue

    image_path = os.path.join(TEST_IMAGES_DIR, image_name)
    label_path = os.path.join(GROUND_TRUTH_DIR, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    # Read ground truth labels
    ground_truth_boxes = read_ground_truth(label_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Run YOLO inference on GPU
    results = model(image, device=DEVICE)

    # Extract YOLO predictions
    predicted_boxes = []
    for result in results:
        for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            predicted_boxes.append((box[0], box[1], box[2], box[3], int(class_id), score))

    # Matching predictions with ground truth using IoU
    matched = set()
    for pred_box in predicted_boxes:
        pred_coords = pred_box[:4]
        best_iou = 0
        best_match = None

        for gt_box in ground_truth_boxes:
            gt_coords = gt_box[:4]
            iou = compute_iou(pred_coords, gt_coords)
            if iou > best_iou:
                best_iou = iou
                best_match = gt_box

        # If IoU is greater than threshold, it's a TP
        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched.add(best_match)
        else:
            fp += 1  # False positive (incorrect detection)

    # False Negatives (missed ground truth boxes)
    fn += len(ground_truth_boxes) - len(matched)

# Compute final metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

# Print results
print("\nEvaluation Results:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
