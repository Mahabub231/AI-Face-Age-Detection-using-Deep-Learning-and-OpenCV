# 🧠 AI FaceAge — Deep Learning Age Detection

Face age detection web application using OpenCV DNN, Caffe models, and PyTorch UTKFace model with single-face and multi-face support.

---

## 🚀 Local Setup (Windows/Mac/Linux)

### 1. Clone Repository
git clone https://github.com/Mahabub231/AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV.git
cd AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV

### 2. Create Virtual Environment
python -m venv .venv

### Activate Environment

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run Application
python app.py

Open in browser:
http://localhost:5000

---

## ☁️ Online Deployment (Render.com)

Set environment variables:

SECRET_KEY=any_random_secret  
AGE_AI_MODE=opencv  
DEBUG=False  

---

## 🤖 AI Modes

opencv  → Fast, Good  
pytorch → Slower, Better  

---

## ✨ Features

- Face detection using OpenCV DNN  
- Age prediction (PyTorch + Caffe fallback)  
- Gender detection  
- Multi-face support  
- Flask web UI  

---

## 📁 Project Structure

AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV/

app.py  
config.py  
models.py  
predict_fixed.py  
download_opencv_models.py  

requirements.txt  
Procfile  
render.yaml  
runtime.txt  

models/  
models_opencv/  
static/  
templates/  

---

## 👨‍💻 Author

Md Mahabub Hasan Mahin
