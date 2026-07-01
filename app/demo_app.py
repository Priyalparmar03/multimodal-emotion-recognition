import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer
from src.models.classifier import MultimodalEmotionModel
from rag.retriever import generate_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ correct 5-class label map from data_prep.py output
label2id = {
    'very_positive': 0,
    'positive': 1,
    'neutral': 2,
    'negative': 3,
    'very_negative': 4
}
id2label = {v: k for k, v in label2id.items()}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = MultimodalEmotionModel(num_classes=5).to(device)
    model.load_state_dict(torch.load(
        "experiments/checkpoints/baseline_model.pt",
        map_location=device
    ))
    model.eval()
    return model

model = load_model()

st.title("Multimodal Emotion Recognition Demo")
st.write("Upload a meme image and enter its text to predict the emotion.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_input("Enter text / caption")

if uploaded_image and text_input:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    encoded = tokenizer(
        text_input,
        padding='max_length', truncation=True, max_length=32,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        pred_id = logits.argmax(1).item()
        predicted_emotion = id2label[pred_id]

    st.subheader(f"Predicted emotion: {predicted_emotion}")

    response = generate_response(predicted_emotion)
    st.text_area("RAG Response", response, height=150)