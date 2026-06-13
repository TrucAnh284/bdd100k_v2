# BDD100K Day-Night Object Detection Project

This repository contains code and configuration files for preparing and evaluating object detection datasets based on BDD100K, with a focus on day-to-night domain adaptation and YOLOv11-based vehicle perception.

## Project Overview

The goal of this project is to study how object detection performance changes under different visual conditions, especially between daytime and nighttime driving scenes. The project prepares multiple YOLO-format datasets and supports experiments on baseline, real-night, and augmented day-to-night data.

The project includes scripts for:

* Cleaning empty YOLO label files
* Counting dataset samples
* Preparing multiple YOLO-format datasets
* Training YOLOv11 models on baseline and augmented datasets
* Validating trained models on real nighttime data
* Comparing detection performance across different data settings

## Repository Structure

```text
BDD100K/
├── cleanup_empty_labels.py
├── count.py
├── prepare_three_datasets.py
├── prepare_ultimate_dataset.py
├── requirements.txt
├── dataset_augmented/
│   ├── dataset.yaml
│   ├── train_yolo11m_ultimate.py
│   └── val_yolo11m_night_real.py
├── dataset_baseline/
│   ├── dataset.yaml
│   ├── README.md
│   ├── train_yolo11m_baseline.py
│   └── val_yolo11m_baseline.py
├── dataset_day_yolo/
├── dataset_night/
├── dataset_night_yolo/
└── outputs/
```

Large files such as images, labels, model weights, training runs, and zip archives are not tracked in GitHub. They are ignored using `.gitignore` to keep the repository lightweight.

## Dataset Settings

This project works with several dataset versions:

| Folder                | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| `dataset_baseline/`   | Baseline YOLO dataset used for initial training and evaluation                |
| `dataset_augmented/`  | Augmented dataset designed for improved robustness under nighttime conditions |
| `dataset_day_yolo/`   | Daytime BDD100K samples converted to YOLO format                              |
| `dataset_night_yolo/` | Nighttime BDD100K samples converted to YOLO format                            |
| `dataset_night/`      | Real nighttime dataset used for validation or comparison                      |

The dataset folders may contain images, labels, training runs, and model weights locally. These files are intentionally excluded from GitHub due to their large size.

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

To prepare the YOLO-format datasets, run:

```bash
python prepare_three_datasets.py
```

or:

```bash
python prepare_ultimate_dataset.py
```

To remove empty label files:

```bash
python cleanup_empty_labels.py
```

To count dataset samples:

```bash
python count.py
```

## Training

Train the baseline YOLOv11m model:

```bash
python dataset_baseline/train_yolo11m_baseline.py
```

Train the augmented YOLOv11m model:

```bash
python dataset_augmented/train_yolo11m_ultimate.py
```

## Validation

Validate the baseline model on the target validation set:

```bash
python dataset_baseline/val_yolo11m_baseline.py
```

Validate the augmented model on real nighttime data:

```bash
python dataset_augmented/val_yolo11m_night_real.py
```

## Notes on Large Files

The following files and folders are not uploaded to GitHub:

```text
images/
labels/
outputs/
runs/
*.zip
*.pt
*.pth
*.onnx
dataset_*/images/
dataset_*/labels/
dataset_*/runs/
dataset_*/*.pt
```

This prevents large datasets and model checkpoints from exceeding GitHub storage limits.

## Research Direction

This project supports experiments related to autonomous driving perception under domain shift. In particular, it focuses on improving object detection performance when moving from daytime driving scenes to nighttime driving scenes.

Main research questions include:

* How does a YOLOv11 detector trained on daytime data perform on nighttime data?
* Can augmented day-to-night data improve detection robustness?
* How much performance difference exists between baseline, real-night, and augmented training settings?
* Which dataset preparation strategy gives the best trade-off between accuracy and data cost?