import os
import io
from PIL import Image,ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
#from transformers import AutoModel
import streamlit as st
import torch
import requests

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))  # Convert bytes data to PIL image
        return image
    else:
        raise FileNotFoundError("No file uploaded")

#Streamlit App
st.set_page_config(page_title="Image Detection")
st.header("Object Detection Application")
#model=AutoModel('./detr-resnet-50')
processor = AutoImageProcessor.from_pretrained('./detr-resnet-50')
model = AutoModelForObjectDetection.from_pretrained('./detr-resnet-50')
#Upload an image
uploaded_file = st.file_uploader("choose an image...", type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
submit = st.button("Detect Objects ")
if submit:
    image_data = input_image_setup(uploaded_file)
    st.subheader("The response is..")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Draw bounding boxes on the image
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        label_text = f"{model.config.id2label[label.item()]} ({round(score.item(), 2)})"
        draw.text((box[0], box[1]), label_text, fill="red")
    
    st.image(drawn_image, caption="Detected Objects", use_column_width=True)
    st.subheader("List of Objects:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        st.write(
            f"Detected :orange[{model.config.id2label[label.item()]}] with confidence "
            f":green[{round(score.item(), 3)}] at location :violet[{box}]"
        )
