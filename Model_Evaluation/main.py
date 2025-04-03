import os
from ultralytics import YOLO

def run_evaluation():
    # Set paths
    MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt"
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    OUTPUT_DIR = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/Model_Evaluation/evaluation_results"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load YOLO model and run validation
        model = YOLO(MODEL_PATH)
        results = model.val(data=DATA_YAML_PATH, save_txt=True, project=OUTPUT_DIR)

        # Extract accuracy metrics
        accuracy = results.box.map50  # mAP@50 (Mean Average Precision at 0.50 IoU)
        accuracy_percentage = accuracy * 100  # Convert to percentage
        
        print(f"Evaluation complete. Results are saved in {OUTPUT_DIR}")
        print(f"Model Accuracy (mAP@50): {accuracy_percentage:.2f}%")

    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == '__main__':
    run_evaluation()
