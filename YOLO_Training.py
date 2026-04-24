import tensorflow as tf
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
import os


image_dir = "YOLO_HAND/TRAIN/images/"
csv_path = "YOLO_HAND/TRAIN/LABELS/YOLO"
output_dir = "Yolo_Data.yaml"


train_images_dir = "YOLO_HAND/TRAIN/images/"
val_images_dir = ("YOLO_HAND/TEST/images/")

yolo_model = YOLO('yolov8n.yaml')

# Define the data.yaml file content dynamically
class_names = ["hand"]
data_yaml_content = f"""
train: {train_images_dir}
val: {val_images_dir}
nc: {len(class_names)}
names: {class_names}
"""

data_yaml_path = os.path.join(output_dir)
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

yolo_model.train(data=data_yaml_path, epochs=50, imgsz=640)
