import os
import io
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
import streamlit as st
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Authenticate with Google Drive using service account credentials
def authenticate():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            "service_account_key.json",
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None

# Function to read image files from a folder
def read_images_from_folder(folder_id, drive_service):
    images = []
    folder_files = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        pageSize=10
    ).execute()
    items = folder_files.get('files', [])
    for item in items:
        filename = item['name']
        file_id = item['id']
        image_bytes = drive_service.files().get_media(fileId=file_id).execute()
        images.append((filename, image_bytes))
    return images

# Function to detect objects in an image using model
def detect_objects_in_image(image_bytes):
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
    image_tensor = image_transform(image).unsqueeze(0)

    # Pass the preprocessed image tensor to the model
    outputs = model(pixel_values=image_tensor)

    # Post-process the outputs
    results = processor.post_process_object_detection(outputs, threshold=0.9)[0]
    detected_objects = [model.config.id2label[label.item()] for label in results["labels"]]
    return detected_objects

# Streamlit App
st.set_page_config(layout="wide")
st.header(r":violet[$\textsf{\large Object Detection From Google Drive}$]")
#st.markdown(":green[Note: Please open the folder which you want to select copy folder-id from the URL]")
#st.markdown(":blue[For Example your folder URL is https://drive.google.com/drive/folders/1g3kBUiESQH15hS7HuMGdefU1B87znRfM then copy 1g3kBUiESQH15hS7HuMGdefU1B87znRfM as folder ID ]")
st.markdown("Please select the folder from your Google Drive:")
st.markdown("[Open Google Drive](https://drive.google.com/drive/my-drive)")


# Load model
processor = AutoImageProcessor.from_pretrained('./detr-resnet-50')
model = AutoModelForObjectDetection.from_pretrained('./detr-resnet-50')

# Choose the folder containing images from Google Drive
folder_id = st.text_input(r":orange[$\textsf{\Large Enter Google Drive folder ID:}$]")
if folder_id.strip():
    submit = st.button(":orange[Detect Objects]")
    if submit:
        drive = authenticate()
        if drive:
            images = read_images_from_folder(folder_id, drive)
            detected_objects = []
            for filename, image_bytes in images:
                objects = detect_objects_in_image(image_bytes)
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
    st.warning("Please enter a valid Google Drive folder ID.")

