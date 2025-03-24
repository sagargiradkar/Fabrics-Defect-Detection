import os
from ultralytics import YOLO

def run_evaluation():
    # Set paths
    MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train2/weights/best.pt"
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    OUTPUT_DIR = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/Prediction/evaluation_results"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load YOLO model and run validation
        model = YOLO(MODEL_PATH)
        results = model.val(data=DATA_YAML_PATH, save_txt=True, project=OUTPUT_DIR)
        print(f"Evaluation complete. Results are saved in {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == '__main__':
    run_evaluation()
