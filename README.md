# YOLOv5: Detecting Blood Cells

## Overview

This project demonstrates how to train and deploy a YOLOv5 model for detecting and classifying blood cells, including white blood cells (WBC), red blood cells (RBC), and platelets. The workflow leverages a publicly available dataset and Google Colab for training, testing, and evaluating the model.


the direct colab link to thenotebook is here :
 - **Colab Notebook:** [Colab Notebook](https://colab.research.google.com/drive/1ywBdVlPd8IpGIEh23z1q6JbdhXAM9qQS?usp=sharing)
The project resources can be accessed via the following public drive 

- **Google Drive:** [Project Resources](https://drive.google.com/drive/folders/13bZkfdJ7bX8VAwSGjUYwNK4ULrkm5TJr?usp=sharing)

Or preferably  repository:

- **Github Repo:** [Project Resources](https://github.com/Abmstpha/BloodCells-YOLOv5)



### Resources:
- **Notebook:** Proejct_YoloV5_detecting_blood_cells.ipynb
- **Model Weights:** best.pt
- **Test Video:** video.mp4
- **Test Output:** output.mp4
- **Execution Recording:** yolo.mp4

### Dataset Information

The dataset consists of 364 images annotated across three classes: WBC, RBC, and Platelets. It was originally open-sourced by [cosmicad](https://github.com/cosmicad/dataset) and [akshaylambda](https://github.com/akshaylamba/all_CELL_data) and further processed using Roboflow.

#### Dataset Summary:
- **Number of Images:** 364
- **Number of Labels:** 4888
- **Classes:**
  - WBC (White Blood Cells)
  - RBC (Red Blood Cells)
  - Platelets

![BCCD health](https://i.imgur.com/BVopW9p.png)

Example Image:

![Blood Cell Example](https://i.imgur.com/QwyX2aD.png)

### Use Cases

This dataset is commonly used for:
- Object detection in medical imaging
- Benchmarking object detection models in small datasets
- Research and academic projects

---

## Workflow

The project workflow is divided into several blocks, as detailed below:

### BLOCK 1: Check GPU & Mount
- Check if GPU is available for faster training.
- Mount Google Drive to access the dataset and save outputs.

```python
import torch
from google.colab import drive

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

drive.mount('/content/drive')
print("Drive mounted. Your dataset is accessible at /content/drive/MyDrive/BloodCellDetection")
```

### BLOCK 2: Clone YOLOv5 Repository & Install Dependencies

- Clone the YOLOv5 repository from GitHub.
- Install required dependencies.

```bash
!pip uninstall ultralytics -y
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt -q
print("YOLOv5 cloned and requirements installed successfully!")
```

### BLOCK 3: List Dataset

- Preview the dataset by listing file contents.
- Check train, validation, and test directories.

```bash
!ls "/content/drive/MyDrive/BloodCellDetection"
!ls "/content/drive/MyDrive/BloodCellDetection/train/images" | head -n 5
!ls "/content/drive/MyDrive/BloodCellDetection/valid/images" | head -n 5
!ls "/content/drive/MyDrive/BloodCellDetection/test/images" | head -n 5
```

### BLOCK 4: Prepare Configuration File

- Create or overwrite `data.yaml` to define dataset paths, class count, and class names.

```yaml
train: /content/drive/MyDrive/BloodCellDetection/train/images
val: /content/drive/MyDrive/BloodCellDetection/valid/images

# Number of classes
nc: 3

# Class names
names: ["Platelets", "RBC", "WBC"]
```

### BLOCK 5: Train the Model

- Train YOLOv5 using the blood cell dataset.
- Configure hyperparameters such as number of epochs, batch size, and input size.

```bash
!python train.py \
  --data /content/drive/MyDrive/BloodCellDetection/data.yaml \
  --cfg yolov5s.yaml \
  --weights yolov5s.pt \
  --epochs 50 \
  --batch 16 \
  --img 640 \
  --name single_folder_exp \
  --cache
```

### BLOCK 6: Visualize Training Results

- Display learning curves to verify training stability.

```python
import os
from IPython.display import Image, display

results_png = "/content/yolov5/runs/train/single_folder_exp/results.png"
if os.path.exists(results_png):
    display(Image(filename=results_png, width=800))
else:
    print("Training results not found. Check your run configuration.")
```

### BLOCK 7: Evaluate the Model

- Validate the trained model on the validation dataset.
- Generate performance metrics such as precision, recall, and mAP.

```bash
!python val.py \
    --weights runs/train/single_folder_exp/weights/best.pt \
    --data /content/drive/MyDrive/BloodCellDetection/data.yaml \
    --img 640 \
    --conf 0.001 \
    --iou 0.65 \
    --task val
```

### BLOCK 8: Display Validation Metrics

- Display validation plots including confusion matrix and precision-recall curve.

```python
import glob
from IPython.display import Image, display

val_dirs = sorted(glob.glob("runs/val/exp*"))
if not val_dirs:
    print("Validation results not found.")
else:
    latest_val_dir = val_dirs[-1]
    visuals = ["F1_curve.png", "PR_curve.png", "confusion_matrix.png", "val_results.png"]
    for img_name in visuals:
        path = os.path.join(latest_val_dir, img_name)
        if os.path.exists(path):
            display(Image(filename=path, width=600))
```

### BLOCK 9: Test the Model

- Run YOLOv5 detection on test images.
- Save annotated outputs in a specified directory.

```bash
!python detect.py \
  --weights runs/train/single_folder_exp/weights/best.pt \
  --source "/content/drive/MyDrive/BloodCellDetection/test/images" \
  --img 640 \
  --conf 0.25 \
  --name single_folder_detect
```

### BLOCK 10: Test on Video

- Detect blood cells in a test video and save the output as an annotated video.

```python
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.plots import Annotator, colors
import torch

video_path = '/content/video.mp4'
out_path = 'output.mp4'
weights_path = '/content/yolov5/best.pt'

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
model = DetectMultiBackend(weights_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = frame[:, :, ::-1].copy()
    results = model(img)
    detections = non_max_suppression(results)
    
    for det in detections:
        for *box, conf, cls in det:
            Annotator(frame).box_label(box, f"{cls} {conf:.2f}", color=colors(cls, True))
    out.write(frame)

cap.release()
out.release()
print("Output video saved as output.mp4")
```

---

## Results

- **Test Images:** Detection results can be found in the `runs/detect/single_folder_detect` folder.
- **Test Video Output:** Output saved as `output.mp4`.

---

## Conclusion

This project showcases the full pipeline for training, validating, and testing a YOLOv5 model on a medical imaging dataset. The results demonstrate that YOLOv5 is a powerful tool for object detection tasks, even with small datasets.
