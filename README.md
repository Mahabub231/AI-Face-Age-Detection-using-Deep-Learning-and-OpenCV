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

| Mode    | Speed   | Accuracy |
|---------|---------|----------|
| opencv  | Fast    | Good     |
| pytorch | Slower  | Better   |

Default mode = opencv (no GPU required)

---

## ✨ Features

- Face detection using OpenCV DNN  
- Age prediction (PyTorch + Caffe fallback)  
- Gender detection  
- Multi-face support  
- Annotated output image  
- Face crop preview  
- Flask web UI  

---

## 📁 Project Structure


AI-Face-Age-Detection-using-Deep-Learning-and-OpenCV/
│
├── app.py # Main Flask app (routes, API, UI handling)
├── config.py # App configuration & environment settings
├── models.py # Database models (SQLAlchemy)
├── predict_fixed.py # Core AI engine (face detection + age/gender)
├── download_opencv_models.py # Script to download OpenCV models
│
├── requirements.txt # Python dependencies
├── Procfile # Deployment config (Render/Heroku)
├── render.yaml # Render deployment setup
├── runtime.txt # Python version for deployment
├── README.md # Project documentation
│
├── models/ # PyTorch trained models
│ └── best_utkface_model.pth
│
├── models_opencv/ # OpenCV pre-trained models
│ ├── opencv_face_detector.pb
│ ├── opencv_face_detector.pbtxt
│ ├── age_net.caffemodel
│ ├── age_deploy.prototxt
│ ├── gender_net.caffemodel
│ └── gender_deploy.prototxt
│
├── static/ # Static files (CSS, JS, uploads)
│ └── uploads/
│ └── crops/ # Detected face images
│
├── templates/ # HTML templates (frontend UI)
│
└── pycache/ # Python cache (ignored)
---

## ⚠️ Important Notes

- .venv → Do NOT upload  
- .env → Do NOT upload  
- Large model files (>100MB) → Use Git LFS  

---

## 👥 Team — Green University of Bangladesh (CSE-404)

| Name | Student ID |
|------|-----------|
| Md Raisul Islam | 231902037 |
| Md Mahabub Hasan Mahin | 231902056 |
| Chinmoy Debnath | 231902029 |

---
## 👨‍💻 Author

Md Mahabub Hasan Mahin

---

## ⭐ Future Improvements

- Better age prediction accuracy  
- Live camera detection  
- Faster inference  
- Mobile optimization  

