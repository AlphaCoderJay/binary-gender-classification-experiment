import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from model import CNN

model = CNN()
checkpoint = torch.load("genderecog_model.pth",map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_names = ["Female","Male"]

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

st.title("👤 Gender Recognizer")
st.markdown("Upload a image of person to find out their gender")

uploaded_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

#When we upload a file
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,caption="Uploaded Image",use_container_width=True)

    input_tensor = transform_test(image).unsqueeze(0)
    
    #Predict
    with torch.no_grad():
        output = model(input_tensor)
        _,pred = torch.max(output,1)
        
        prediction = class_names[pred.item()]

    st.success(f"Predicted Gender {prediction}**")