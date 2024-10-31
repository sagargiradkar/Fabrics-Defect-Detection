---

# Automated E-Waste Sorting with Cobots and Computer Vision

## Overview

This project aims to develop an automated system for sorting electronic waste (e-waste) using collaborative robots (cobots) and advanced computer vision techniques. The system identifies and sorts various e-waste components, such as smartphone batteries, motherboards, lenses, and microphones, improving recycling efficiency and promoting sustainable e-waste management.

## Project Structure

### Key Components

1. **Input System**: A conveyor belt feeds e-waste items into the sorting area for processing.
2. **Sensing System**: Vision and capacitive sensors detect and analyze the components of e-waste.
3. **Processing System**: A Raspberry Pi paired with a GPU processes sensor data using deep learning and machine learning (ML) algorithms.
4. **Robotic System**: A cobot equipped with interchangeable grippers handles the picking and sorting of e-waste items.
5. **Control System**: Robot Operating System (ROS) facilitates communication between components and controls the cobot’s movements.
6. **Output System**: The sorted e-waste items are placed into designated bins for further processing or recycling.

The system operates in a closed-loop, continually learning and optimizing its sorting accuracy.

### Files in Repository

- **`.gitignore`** - Specifies intentionally untracked files to ignore.
- **`live_streaming.py`** - Script to enable live streaming of the sorting process.
- **`live_streaming_7000.py`** - Alternative live-streaming script for a different port or configuration.
- **`static_images.py`** - Processes e-waste detection on static images.
- **`static_images_7000.py`** - Alternate script for processing static images with a different configuration.
- **`train.py`** - Training script to develop the e-waste sorting model.
- **`train7000.py`** - Alternate training script with different parameters or dataset size.

## Getting Started

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `train.py` or `train7000.py` to train the model on your dataset.
4. Use `live_streaming.py` or `static_images.py` to test the system on live or static data.

---

