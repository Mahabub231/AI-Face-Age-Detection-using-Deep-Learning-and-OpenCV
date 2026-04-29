# 🧠 AI FaceAge — Deep Learning Age Detection

Face age detection web application using OpenCV DNN, Caffe models, and PyTorch UTKFace model with single-face and multi-face support.

## 🚀 Local Setup

git clone https://github.com/Mahabub231/AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV.git
cd AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt

python app.py

# Open in browser
http://localhost:5000

## ☁️ Render Deployment Environment Variables

SECRET_KEY=any_random_secret
AGE_AI_MODE=opencv
DEBUG=False
