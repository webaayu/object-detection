import os
import io
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Load model and setup Streamlit
processor = AutoImageProcessor.from_pretrained('./detr-resnet-50')
model = AutoModelForObjectDetection.from_pretrained('./detr-resnet-50')
st.set_page_config(page_title="Image Detection")

# Function to read image files from a folder
def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            images.append((filename, image_path))
    return images

# Function to detect objects in an image using the specified model


def detect_objects_in_image(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Convert bytes data to PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and normalize the image
    image_transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = image_transform(image).unsqueeze(0) #image_tensor has four parameters (numbar of image in batch,height,width,'rgb')

    # Pass the preprocessed image tensor to the model
    outputs = model(pixel_values=image_tensor)  # model accepts pixel_values directly

    # Post-process the outputs
    results = processor.post_process_object_detection(outputs, threshold=0.9)[0]
    detected_objects = [model.config.id2label[label.item()] for label in results["labels"]]
    return detected_objects

# Streamlit App
st.header("Object Detection Application")

# Choose the folder containing images
folder_path = st.text_input("Enter folder path containing images:")
if os.path.isdir(folder_path):
    submit = st.button("Detect Objects")
    if submit:
        detected_objects = []
        for filename, image_path in read_images_from_folder(folder_path):
            objects = detect_objects_in_image(image_path)
            detected_objects.append((filename, objects))

        # Save detected objects to a DataFrame and download as CSV
        df = pd.DataFrame(detected_objects, columns=["Image", "Detected Objects"])
        st.write(df)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name='detected_objects.csv',
            mime='text/csv',
        )
else:
    st.error("Invalid folder path. Please enter a valid folder path.")

