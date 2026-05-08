# HoT3D Ego Hand Tracking – Phase 1

**Name:** Jake Digiugno  
**B-Number:** B00972712

## Overview
This project implements baseline models for ego-centric hand pose estimation using the HoT3D dataset. A PyTorch pipeline was developed to load synchronized Project Aria VRS recordings and corresponding hand pose annotations.

Three baseline approaches were evaluated:
- Single-view baseline
- Multi-view average fusion
- Multi-view concatenation fusion

All models use a modified ResNet-18 backbone to regress 22 hand joint angles.

---

## Implemented Methods

### Single-View Baseline
Uses one grayscale camera image to predict hand joint angles.

### Multi-View Average Fusion
Runs predictions independently on left and right camera images and averages the outputs.

### Multi-View Concatenation
Extracts features from both camera views, concatenates them, and predicts joint angles jointly.

---

## Results

| Model | Best Validation Loss |
|---|---|
| Single-view | 0.0383 |
| Multi-view average | 0.0394 |
| Multi-view concatenation | 0.0391 |

---

## Dataset Pipeline

The dataset loader:
- Parses hand pose annotations from JSONL files
- Loads synchronized frames from VRS recordings
- Aligns timestamps between annotations and images
- Supports both single-view and multi-view modes

Images are:
- converted to grayscale
- resized to 224×224
- converted to PyTorch tensors

---

## Training Setup

- Optimizer: Adam
- Learning rate: 1e-3
- Loss function: Mean Squared Error (MSE)
- Batch size: 4
- Epochs: 10
- Train/Validation split: 80/20

---

## Running the Code

Enter the Pixi environment:

```bash
pixi shell
```

Run training:

```bash
python train.py
```

Change modes inside `train.py`:

```python
MODE = "single"
MODE = "multi_avg"
MODE = "multi_concat"
```

---

## Project Structure

```text
dataset/
    loader.py

models/
    baseline_model.py
    multiview_model.py

train.py
README.md
```

---

## Notes

This project uses:
- PyTorch
- torchvision
- Project Aria Tools
- HoT3D dataset