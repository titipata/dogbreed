import streamlit as st
import json
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_to_idx = json.load(open("class_to_idx.json", "r"))
idx_to_class = {v: k for k, v in class_to_idx.items()}
n_classes = len(idx_to_class.keys())  # number of breeds classes

# Note that I only save `fc` layer weights, and not the whole model.
# model was trained and the linear classifier is saved as follows:
# torch.save(model.fc.state_dict(), "fc.pt")
@st.cache(suppress_st_warning=True)
def load_model():
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes)
    )
    MODEL_PATH = "fc.pt"
    model.fc.load_state_dict(torch.load(MODEL_PATH))
    return model

model = load_model()

@st.cache(suppress_st_warning=True)
def predict(path: str):
    """Predict from a given path"""
    img = Image.open(path)
    img = transform(img)
    model.eval()
    logits = model(img.unsqueeze(0))
    pred = logits.argmax().tolist()
    return pred


st.title("Dog Breed Classification")
st.write("""
🐶🐶🐶 Upload your dog image (or even yourself) to see what breed it is. 🐶🐶🐶
""")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.caption("Classifying...")
    pred = predict(uploaded_file)
    st.write(f'🐕 Predicted breed: **{idx_to_class[pred].capitalize().replace("_", " ")}**')
