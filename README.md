# Fabric Defect Detection using YOLOv11

## Overview
This project focuses on detecting fabric defects using the YOLOv11 model. The model has been trained using a labeled dataset to identify three types of defects: **Hole**, **Stitch**, and **Seam**.

## Dataset
- **Dataset URL:** [Roboflow FinalRun - Version 5](https://universe.roboflow.com/thesis-wy7ne/finalrun/dataset/5)
- **Training Images:** 8,970
- **Validation Images:** 204
- **Test Images:** 882
- **Classes:**
  - Hole
  - Stitch
  - Seam

## Project Structure
```
Fabrics-Defect-Detection/
├── model_training/
│   ├── runs/detect/train2/weights/best.pt  # Best model weights
├── dataset_8970/
│   ├── train/                               # Training images and labels
│   ├── valid/                               # Validation images and labels
│   ├── test/                                # Test images and labels
├── evaluation_results/                     # Evaluation outputs
└── Prediction/
    ├── main.py                              # Evaluation script
```

## Requirements
Ensure you have the following packages installed:

```bash
pip install ultralytics torch
```

## Evaluation
To evaluate the model, run the following command:

```bash
cd Prediction
python main.py
```

The results will be saved in the `evaluation_results` folder.

## Troubleshooting
- Ensure the model path and dataset paths in `main.py` are correct.
- If using Windows, ensure `if __name__ == '__main__':` is used to avoid multiprocessing issues.

## Contact
For further assistance, feel free to reach out or raise an issue on the repository.

