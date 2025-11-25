<img width="1" height="1" alt="image" src="https://github.com/user-attachments/assets/26e35f22-2fea-467a-be9c-62853d67e28b" />🖊️ Signature Detection Using Deep Learning
----------------------------------------------------------------------------------------------------------------------
A lightweight, PyTorch-based machine learning project that detects whether an uploaded image contains a handwritten signature. The model is trained on a custom dataset of signature and non-signature images and deployed using Streamlit.

⭐ Project Overview
----------------------------------------------------------------------------------------------------------------------
This project identifies the presence of a signature in an image using a custom Convolutional Neural Network (CNN).
It includes:

Dataset preprocessing

Model training

Model evaluation

A Streamlit web application for real-time inference

A clean deployment-ready architecture

🧠 Tech Stack Used:
----------------------------------------------------------------------------------------------------------------------
Machine Learning / Deep Learning

PyTorch — Model building, training, inference

Torchvision — Image transformations

CNN Architecture — Custom SignatureNet model

Preprocessing Tools

OpenCV — Image resizing + cleaning

Pillow (PIL) — Image loading

NumPy — Array operations

Deployment

Streamlit — Web app

GitHub — Code hosting

🚀 How It Works
----------------------------------------------------------------------------------------------------------------------
1️⃣ Image Preprocessing

Uploaded images are resized to 128×128, normalized, and converted to tensors.

2️⃣ CNN-based Prediction

The image passes through a trained deep learning model that outputs a probability (0–1).

3️⃣ Result Display

    If the probability exceeds a threshold (default 0.50), the app declares:

     _Signature Detected_

    Otherwise:

     _No Signature Found_

▶️ Running the Project Locally

Install Dependencies
       
    pip install -r requirements.txt

Run the App

    streamlit run app.py

🧪 Model Training (Optional)
----------------------------------------------------------------------------------------------------------------------
👉If you want to retrain the model:

    python src/prepare_dataset.py
 
    python src/train_upgraded.py

    python src/evaluate.py



👉 Make sure your dataset follows:

    data/raw/positive

    data/raw/negative

📌 Features
----------------------------------------------------------------------------------------------------------------------
Lightweight & fast

Works on any RGB image

Trained on clean signature datasets

Adjustable confidence threshold

Mobile & web-friendly interface

----------------------------------------------------------------------------------------------------------------------

🙋‍♀️ Author

Bora Keerthi Sri Reddy

BTech CSE | Web Developer | Tech Enthusiast
