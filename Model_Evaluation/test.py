import os
import subprocess

def evaluate_yolo(model_path, data_yaml, conf_thresh=0.25, iou_thresh=0.5):
    try:
        # Command to run YOLOv11 evaluation using Python module
        command = [
            'python', '-m', 'ultralytics', 'yolo',
            'task=detect',
            'mode=val',
            f'model={model_path}',
            f'data={data_yaml}',
            f'conf={conf_thresh}',
            f'iou={iou_thresh}'
        ]
        
        # Execute the command
        subprocess.run(command, check=True)
        print("Evaluation completed. Check the results in 'runs/detect/val/'.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    model_path = 'C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt'
