# Object Tracking Competition

## Overview

Participants must implement an object tracker that:

- takes a video sequence as input
- predicts one bounding box per frame
- outputs predictions in CSV format

The organizer evaluation script automatically:

- loads videos
- measures latency
- saves predictions
- validates submission format

Participants only modify:

- `predictor.py`

---

# Repository Structure

```text
submission/
├── inference.py
├── predictor.py
├── download.py
├── check_submission.py
├── requirements.txt
├── sample_submission.csv
└── checkpoints/
```

---

# Installation

Create environment:
 requirements should include the exact versions not just the libraries
```bash
pip install -r requirements.txt
```

---

# Step 1 — Download Checkpoints

Download your model weights:

```bash
python download.py
```

This should download checkpoints into:

```text
checkpoints/
```

Example:

```text
checkpoints/model.pth
```
> You are required to replace "YOUR_FILE_ID" with the actual file ID from Google Drive to download the checkpoint.
Or replace the download method with your preferred method if you are not using Google Drive.
>
> If we cannot download the file successfully, your team will be disqualified 
---

# Step 2 — Run Inference

Run tracking inference:

```bash
python inference.py \
    data/test.json \
    split_name \
    predictions.csv
```

Arguments:

```text
1. input json
2. split name
3. output csv
```


---

# Step 3 — Validate Submission

Check that your submission format is correct:

```bash
python check_submission.py \
    sample_submission.csv \
    predictions.csv
```

This verifies:

- correct CSV columns
- correct frame IDs
- correct number of predictions

---

# Required CSV Format

Your predictions must follow:

```csv
id,x,y,w,h
dataset1/Car_video_0,0,0,0,0
dataset1/Car_video_1,0,0,0,0
dataset1/Car_video_2,0,0,0,0
```

---

# Required Output

Your tracker must return:

```python
[
    {
        "frame_idx": 0,
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
    }
]
```

One prediction per frame.

---

# Rules

## Allowed

- PyTorch
- OpenCV
- Any tracking architecture
- Any Python libraries in `requirements.txt`

## Not Allowed

- Absolute paths
- Interactive input
- Manual file selection
- Modifying `inference.py`

---

# Notes

- The first-frame bounding box is provided.
- One bounding box must be predicted for every frame.
- Relative paths only.
- The evaluation environment may not have internet access during inference.

---

# Another reminder this should  be what we will do (any failure in this will lead to immediate disqualification) 

```bash
# install dependencies
pip install -r requirements.txt

# download checkpoint
python download.py

# run inference
python inference.py \
    data/test.json \
    hidden \
    predictions.csv

# validate predictions
python check_submission.py \
    sample_submission.csv \
    predictions.csv
```# NewbieSquad - UAV Object Tracking (AIC-4 Phase I)

This repository contains our submission for the MTC AIC-4 Phase I UAV Tracking Competition.
Our approach utilizes a pre-trained UETrack model with heavily optimized inference-time algorithms specifically designed for erratic UAV motion, extreme scale changes, and severe occlusions.

## Directory Structure
- `checkpoints/`: Directory where the pre-trained `model_final.pth` (uetrack_base) should be placed.
- `inference_scripts/`: Contains `inference.py` for evaluating on the test set.
- `training_scripts/`: Contains notes on training.
- `tests/`: Contains verification scripts and automated tests.
- `paper/`: Contains the PDF of our system description.

---

## Model Weights
Due to GitHub's file size limits, the pre-trained model checkpoint (`model_final.pth`) is hosted externally. 
Please download the weights from our **[Google Drive Folder](https://drive.google.com/drive/folders/18uOU8gPKn1ejLtfVWgncdKaGvbkavUjC?usp=sharing)** and place the file directly inside the `checkpoints/` directory before building the Docker image or running inference.

## Precise Commands for Verification

### 1. Model Training
Our solution utilizes a **zero-shot** approach leveraging the pre-trained UETrack weights. We did not perform any additional fine-tuning or training on the competition data. Therefore, no training commands are required. Ensure you have downloaded the weights into the `checkpoints/` folder as instructed above.

### 2. Model Inference
To execute inference in an isolated Docker environment exactly as required by the competition specifications, follow these commands from the root of this repository:

**Step A: Build the Docker Image**
```bash
docker build -t newbiesquad_submission .
```

**Step B: Run Inference (Air-Gapped)**
*Note: Replace `/your/local/test/data` with the path to the hidden test set, and `/your/local/output` with the directory where `submission.csv` should be saved.*

```bash
docker run --rm --gpus all \
    --network none \
    -v /your/local/test/data:/workspace/data:ro \
    -v /your/local/output:/workspace/mtc_uav_uetrack \
    newbiesquad_submission
```

The script will process the `contestant_manifest.json` located at `/workspace/data` and successfully output the `submission.csv` to your designated output folder.
