# ComputerVisionProject

# Overview

This repository contains code for training and evaluating object detection models using YOLOv8 and YOLOv11, seen in the YOLO_training notebook. Additionally, the repo contains an experimental ViT.ipynb notebook where we began our testing with Vision Transformer models. However, the vision transformer fine tuning and testing was halted due to the high compute and VRAM requirements of even the smallest models. 

# Requirements

- Recent Python Version 
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library
- Downloading the Dataset
- Creating data.yaml file (included in the repo)
- Graphics card is highly recomended 

# Dataset

We are using the "C2A: Combination to Application Dataset:, which is a synthetic dataset of UAV imagery for human detection in disaster scenarios. 
The dataset can be downloaded from the GitHub page below:  
🔗 **[Dataset GitHub page](https://github.com/Ragib-Amin-Nihal/C2A)**

Below is the structure of the dataset as described by the GitHub repo:
```
new_dataset3/  
│  
├── All labels with Pose info/  
│   └── [YOLO format labels with pose information for all images]  
│  
├── test/  
│   ├── images/  
│   │   └── [Test image files]  
│   ├── labels/  
│   │   └── [YOLO format label files for test images]  
│   └── test_annotations.json  [COCO format annotations for test set]  
│  
├── train/  
│   ├── images/  
│   │   └── [Training image files]  
│   ├── labels/  
│   │   └── [YOLO format label files for training images]  
│   └── train_annotations.json  [COCO format annotations for training set]  
│  
└── val/  
    ├── images/  
    │   └── [Validation image files]  
    ├── labels/  
    │   └── [YOLO format label files for validation images]  
    └── val_annotations.json  [COCO format annotations for validation set]
```
### YAML File Configuration
The data.yaml file is required for finetuning the YOLO model using the Ultralytics library. It tells the library where the dataset is stored and its structure. Below is the default yaml file for this dataset, you can change it to point to a preferred path.

The dataset configuration file (`data.yaml`) should look like this:

```yaml
path: ./C2A_Dataset/new_dataset3
train: train/images
val: val/images
test: test/images  

nc: 1              
names: ['human'] 
```

We stored this file inside the same directory as the dataset, and pointed to it inside the python code.

```python
data='./C2A_Dataset/new_dataset3/data.yaml'
```

---

# Step-by-Step Instructions

## 1. Preparing the Dataset
First download the dataset from the provided link, and extract it to a suitable directory. Then edit the provided YAML file as needed. 


## 2. Creating pose estimation labels 

To create separate labels directory that has Yolo labels with pose information for Train/Validation/Testing split, run the following code snippet in the YOLO_training notebook:

(This step is needed because the pose labels are not organized into train/val/test in the dataset)
```python
import os

# Define paths
base_dir = "./C2A_Dataset/new_dataset3"
pose_labels_dir = os.path.join(base_dir, "All labels with Pose information", "labels")
sub_dirs = ["train", "val", "test"]

# Iterate through directories and modify labels
for sub_dir in sub_dirs:
    image_dir = os.path.join(base_dir, sub_dir, "images")
    old_label_dir = os.path.join(base_dir, sub_dir, "labels")
    new_label_dir = os.path.join(base_dir, sub_dir, "labels_1")

    # Process label files
    for image_file in sorted(os.listdir(image_dir)):
        image_name = os.path.splitext(image_file)[0]
        pose_label_path = os.path.join(pose_labels_dir, f"{image_name}.txt")
        new_label_path = os.path.join(new_label_dir, f"{image_name}.txt")

        if os.path.exists(pose_label_path):
            with open(pose_label_path, "r") as file:
                lines = file.readlines()

            # Update label formatting
            new_lines = [
                f"{line.split()[-1]} {' '.join(line.split()[1:-1])}\n" for line in lines if len(line.split()) >= 6
            ]

            with open(new_label_path, "w") as file:
                file.writelines(new_lines)
        else:
            print(f"Pose label not found for image: {image_file}")
```

---

## 3.  Training and Testing the models

### **YOLOv8**

#### training

To train the YOLOv8 nano model, Run :

```python
from ultralytics import YOLO

# Initialize YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='./C2A_Dataset/new_dataset3/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    name="yolov8n-custom"
)
```

Note that the batch size is low due to hardware limitations, this can be increased according to available VRAM. 
#### Testing

```python
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO('./runs/detect/yolov8n-custom2/weights/best.pt')

# Evaluate on the test dataset
metrics = model.val(
    data='./C2A_Dataset/new_dataset3/data.yaml',
    split='test',
    imgsz=640,
    batch=16
)

print(metrics)
```

### **YOLOv11** (With pose estimation)

#### Training YOLOv11

Run the following code snippet:

```python
from ultralytics import YOLO

# Create a YOLOv11 Nano model
model = YOLO('yolo11n.pt')  # YOLOv11 Nano pretrained weights

# Train the model
model.train(
    data='./C2A_Dataset/new_dataset3/data.yaml',     # Dataset path and configuration
    epochs=50,            # Number of training epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size for training
    name="yolov11-pose_estemation" # Experiment name

)
```

#### Testing

To evaluate the model with the best checkpoint, run the following snippet:
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('./runs/detect/yolov11-pose_estemation/weights/best.pt')  # Path to the best weights after training

# Evaluate the model on the test dataset
metrics = model.val(
    data='./C2A_Dataset/new_dataset3/data.yaml',  # Path to your dataset configuration
    split='test',                                # Specify the test split
    imgsz=640,                                   # Image size for evaluation
    batch=16                                     # Batch size for evaluation
)

# Print evaluation metrics
print(metrics)
```

### **YOLOv11** (Without pose estimation)

#### Training 

Run the following code snippet:

```python
from ultralytics import YOLO

# Create a YOLOv11 Nano model
model = YOLO('yolo11s.pt')  # YOLOv11 small pretrained weights

# Train the model
model.train(
    data='./C2A_Dataset/new_dataset3/data.yaml',     # Dataset path and configuration
    epochs=50,            # Number of training epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size for training
    name="yolov11_run1" # Experiment name
)
```

#### Testing

To evaluate the model with the best checkpoint, run the following snippet:
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('./runs/detect/yolov11_run1/weights/best.pt')  # Path to the best weights after training

# Evaluate the model on the test dataset
metrics = model.val(
    data='./C2A_Dataset/new_dataset3/data.yaml',  # Path to your dataset configuration
    split='test',                                # Specify the test split
    imgsz=640,                                   # Image size for evaluation
    batch=16                                     # Batch size for evaluation
)

# Print evaluation metrics
print(metrics)
```

---

### Load a model from a checkpoint

To resume training from a particular checkpoint file if the need arises, use the following code snippet to load the checkpoint:

```python
from ultralytics import YOLO

# Load the model from the last checkpoint
model = YOLO('./runs/detect/yolov11_run1/weights/last.pt')  # Path to last trained weights

model.train(
    data='./C2A_Dataset/new_dataset3/data.yaml',  # Dataset path and configuration
    epochs=50,           # Additional training epochs
    batch=16,            # Batch size
    imgsz=640,           # Image size for training
    name="yolov11_run1_1"  # New experiment name
)
```
---

# Demo Video
You can find a demo video of the project in the repo titled "[Demo.mp4](./Demo.mp4)"
