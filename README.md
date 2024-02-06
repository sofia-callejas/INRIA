# Phone Detection

## Objective

The goal of this task is to identify the coordinates of a phone within an image using an object detection algorithm.

## Getting Started

### Dependencies

This project was developed using:

- MacOS
- Python 3.9.6
- Torch 2.0.1
- Ultralytics 8.1.9

### Installation

To set up the virtual environment, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

# Repository Overview

The repository contains essential files and folders:

- **`find_phone.py`**:
  - Script for testing the model's ability to locate phones.

- **`train_phone_find.py`**:
  - Script responsible for training the model to recognize and locate phones.

- **`Inria-Report.html`**:
  - HTML file containing the comprehensive report on the task, findings, and results.

- **`Preprocessing` Folder**:
  - Contains two sub-folders:
    - One with data augmented for training.
    - Another with custom-preprocessed data.

- **`runs` Folder**:
  - Stores the results of the algorithm's executions.

## Training

To train the model with the dataset, execute the following command:

```bash
python train_phone_finder.py ~/find_phone_data
```

# Testing

To test an image and obtain normalized coordinates of the phone, use the following command:

```bash
python find_phone.py ~/find_phone_test_images/51.jpg
```
