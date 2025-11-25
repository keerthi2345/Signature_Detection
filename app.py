import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os
from model import SignatureNet

# Paths
#ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join("weights", "signature_model.pth")

# Model input size (must match training)
IMAGE_SIZE = (128, 128)

# Load model once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignatureNet().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

model, device = load_model()

# Transform for input image
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Streamlit page setup
st.set_page_config(page_title="Signature Detector", layout="centered")
st.title("🖊️ Signature Detector")
st.markdown(
    "Upload an image"
    "The model outputs the probability that the image contains a signature."
)

# Upload
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])

col1, col2 = st.columns([2,1])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    with col1:
        st.image(img, use_container_width=True)
    
    with col2:
        st.write("Settings")
        threshold = st.slider("Signature probability threshold", 0.1, 0.9, 0.5, step=0.05)
        run_btn = st.button("Analyze")

    if run_btn:
        # Preprocess
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()  # safer than indexing

        if prob >= threshold:
            st.success(f"✅ Signature detected — confidence {prob:.2f}")
        else:
            st.error(f"❌ No signature — confidence {prob:.2f}")

        st.caption(
            "Tip: If you get false positives on doodles, add more non-signature images "
            "(scribbles, handwritten words, emojis) to data/processed/negative and retrain."
        )
