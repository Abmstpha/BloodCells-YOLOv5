```markdown
# YOLOv5: Detecting Blood Cells

## Resources

The resources for this project, including the notebook, model weights, videos, and dataset, are provided below:

- **Colab Notebook:** [Colab Notebook](https://colab.research.google.com/drive/1ywBdVlPd8IpGIEh23z1q6JbdhXAM9qQS?usp=sharing)
- **Google Drive:** [Project Resources](https://drive.google.com/drive/folders/13bZkfdJ7bX8VAwSGjUYwNK4ULrkm5TJr?usp=sharing)
- **Github Repo:** [Project Resources](https://github.com/Abmstpha/BloodCells-YOLOv5)

### Contents:
- **Notebook:** `Proejct_YoloV5_detecting_blood_cells.ipynb`
- **Model Weights:** `best.pt`
- **Test Video:** `video.mp4`
- **Test Output:** `output.mp4`
- **Execution Recording:** `yolo.mp4`

---

## Overview

This project showcases the application of YOLOv5 for detecting and classifying blood cells, including white blood cells (WBC), red blood cells (RBC), and platelets. It covers the complete workflow from dataset preparation to training, evaluation, and inference on new data. 

---

## Dataset Information

The dataset comprises 364 images, annotated across three classes. Originally sourced from open repositories ([cosmicad](https://github.com/cosmicad/dataset) and [akshaylambda](https://github.com/akshaylamba/all_CELL_data)), it has been processed further using Roboflow.

### Dataset Summary:
- **Number of Images:** 364
- **Number of Labels:** 4888
- **Classes:**
  - WBC (White Blood Cells)
  - RBC (Red Blood Cells)
  - Platelets

#### Visual Insights
- **Dataset Health Check:**
  ![BCCD health](https://i.imgur.com/BVopW9p.png)

- **Example Image:**
  ![Blood Cell Example](https://i.imgur.com/QwyX2aD.png)

---

# Step-by-Step Workflow

## Step 1: Setting Up the Environment

First, we verify if a GPU is available for training, mount Google Drive to access the dataset, and ensure all resources are accessible.

```python
# Import necessary libraries and check for GPU
import torch
from google.colab import drive

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Mount Google Drive
drive.mount('/content/drive')
print("Drive mounted. Your dataset is accessible at /content/drive/MyDrive/BloodCellDetection")
```

---

## Step 2: Cloning YOLOv5 and Installing Dependencies

Next, clone the YOLOv5 repository from GitHub and install the required dependencies.

```bash
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt -q
```

---

## Step 3: Previewing the Dataset

We explore the dataset structure by listing files from the train, validation, and test directories. This ensures the dataset is correctly set up.

```bash
!ls "/content/drive/MyDrive/BloodCellDetection"
!ls "/content/drive/MyDrive/BloodCellDetection/train/images" | head -n 5
!ls "/content/drive/MyDrive/BloodCellDetection/valid/images" | head -n 5
!ls "/content/drive/MyDrive/BloodCellDetection/test/images" | head -n 5
```

---

## Step 4: Preparing the Configuration File

The `data.yaml` file specifies dataset paths, the number of classes, and class names. Create the following configuration file:

```yaml
train: /content/drive/MyDrive/BloodCellDetection/train/images
val: /content/drive/MyDrive/BloodCellDetection/valid/images

# Number of classes
nc: 3

# Class names
names: ["Platelets", "RBC", "WBC"]
```

---

## Step 5: Training the YOLOv5 Model

Using the blood cell dataset, train YOLOv5 with the following command. Adjust the hyperparameters like epochs and batch size as needed.

```bash
!python train.py \
  --data /content/drive/MyDrive/BloodCellDetection/data.yaml \
  --cfg yolov5s.yaml \
  --weights yolov5s.pt \
  --epochs 50 \
  --batch 16 \
  --img 640 \
  --name blood_cell_exp \
  --cache
```

---

## Step 6: Visualizing Training Results

Once training is complete, YOLOv5 generates results like loss curves, precision, and recall over epochs. Display the training results:

```python
from IPython.display import Image, display

results_png = "/content/yolov5/runs/train/blood_cell_exp/results.png"
if os.path.exists(results_png):
    display(Image(filename=results_png, width=800))
else:
    print("Training results not found.")
```

---

## Step 7: Evaluating the Model

Validate the trained model on the validation dataset to evaluate its performance. Metrics like precision, recall, and mAP are generated.

```bash
!python val.py \
    --weights runs/train/blood_cell_exp/weights/best.pt \
    --data /content/drive/MyDrive/BloodCellDetection/data.yaml \
    --img 640 \
    --conf 0.001 \
    --iou 0.65 \
    --task val
```

---

## Step 8: Running Inference on Images

Use the YOLOv5 detection script to test the model on new images. The annotated results will be saved in the specified directory.

```bash
!python detect.py \
  --weights runs/train/blood_cell_exp/weights/best.pt \
  --source "/content/drive/MyDrive/BloodCellDetection/test/images" \
  --img 640 \
  --conf 0.25 \
  --name blood_cell_test
```

---

## Step 9: Testing the Model on a Video

Finally, apply the model to detect blood cells in a video. The output will be an annotated video.

```python
import cv2
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

    results = model(frame)
    detections = non_max_suppression(results)
    
    for det in detections:
        Annotator(frame).box_label(box, f"{cls} {conf:.2f}", color=colors(cls, True))
    out.write(frame)

cap.release()
out.release()
print("Output video saved as output.mp4")
```

---

## Results

### Test Outputs
- **Test Images:** Annotated detection results can be found in the `runs/detect/blood_cell_test` folder.
- **Test Video Output:** The processed video is saved as `output.mp4`.

### Performance Insights
The training and validation metrics indicate high precision, recall, and F1 scores, demonstrating the model’s ability to accurately detect and classify blood cells.

---

## Conclusion

This project provides a complete workflow for training and deploying YOLOv5 for blood cell detection. With its high accuracy and fast inference, YOLOv5 is a powerful tool for medical imaging tasks. Explore the provided resources to replicate or extend this project for your own datasets!
```