"""
download_opencv_models.py
=========================
Downloads the pre-trained OpenCV Caffe models for face detection + gender.
Run this ONCE before starting the app if models_opencv/ is empty.

Usage:
    python download_opencv_models.py
"""
import urllib.request, os, sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models_opencv"
MODEL_DIR.mkdir(exist_ok=True)

URLS = {
    "opencv_face_detector.pbtxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}

MIN_SIZES = {
    "age_net.caffemodel":    20_000_000,
    "gender_net.caffemodel": 20_000_000,
    "opencv_face_detector_uint8.pb": 2_000_000,
}

def check_missing():
    missing = []
    for name in URLS:
        p = MODEL_DIR / name
        min_sz = MIN_SIZES.get(name, 1024)
        if not p.exists() or p.stat().st_size < min_sz:
            missing.append(name)
    return missing

def download_all(force=False):
    missing = URLS.keys() if force else check_missing()
    if not missing:
        print("✅ All OpenCV models already present.")
        return True

    print(f"⬇️  Downloading {len(list(missing))} model file(s)...\n")
    ok = True
    for name in missing:
        url  = URLS[name]
        path = MODEL_DIR / name
        print(f"  ⬇  {name}")
        try:
            urllib.request.urlretrieve(url, path)
            kb = path.stat().st_size // 1024
            print(f"     ✅ Done ({kb} KB)")
        except Exception as e:
            print(f"     ❌ FAILED: {e}")
            ok = False

    if ok:
        print(f"\n✅ All models saved to: {MODEL_DIR}")
    else:
        print("\n⚠️  Some downloads failed. Retry or download manually.")
    return ok

if __name__ == "__main__":
    force = "--force" in sys.argv
    success = download_all(force=force)
    sys.exit(0 if success else 1)
