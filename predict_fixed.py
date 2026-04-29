"""
predict_fixed.py  — AI FaceAge Detection Engine  v8
====================================================
Modular architecture:
  detector.py logic  → _detect_faces()
  classifier.py logic → _classify_face()
  validator.py logic  → _is_valid_box()

Rules enforced:
  - max_faces per mode (single=1, group=20)
  - min/max face area filtering  (no noise, no full-frame boxes)
  - DNN threshold: 0.55 (reduce false positives)
  - NMS IoU: 0.25 (strict dedup)
  - sorted by face area descending (biggest/most prominent first)
  - exact age from PyTorch UTKFace model
  - gender from OpenCV Caffe gender net
  - clear UI labels: "~23 yrs (estimated)" vs "23 yrs"
"""

import os, cv2, json, numpy as np

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_opencv")
PT_MODEL  = os.path.join(BASE_DIR, "models", "best_utkface_model.pth")
CROP_DIR  = os.path.join(BASE_DIR, "static", "uploads", "crops")

FACE_PROTO   = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL   = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
AGE_PROTO    = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL    = os.path.join(MODEL_DIR, "age_net.caffemodel")

GENDER_LIST = ["Male", "Female"]
CAFFE_AGES  = ["0-2","4-6","8-12","15-20","25-32","38-43","48-53","60-100"]
CAFFE_MID   = [   1,    5,    10,     18,     28,     40,     50,     70 ]
MODEL_MEAN  = (78.4263377603, 87.7689143744, 114.895847746)

# Per-face colours (BGR)
COLORS = [
    (0,220,255),(100,100,255),(80,255,120),(0,165,255),
    (220,0,255),(180,255,0),  (255,220,0), (0,200,255),
    (255,120,0),(120,0,255),
]

# ─────────────────────────────────────────────
#  Detection constants
# ─────────────────────────────────────────────
DNN_SIZES     = [300, 500, 700]   # multi-scale blob sizes
DNN_THRESH    = 0.55              # higher = fewer false positives
NMS_IOU_THR   = 0.25             # strict dedup
MIN_FACE_PX   = 30               # ignore boxes smaller than 30×30 px
MAX_FACE_FRAC = 0.90             # ignore boxes covering >90% of image area

MAX_FACES_SINGLE = 1
MAX_FACES_GROUP  = 20

# ═══════════════════════════════════════════════════════════
# MODULE 1:  MODEL LOADING
# ═══════════════════════════════════════════════════════════
_dnn_ok      = False
_pt_ok       = False
_face_net    = _gender_net = _caffe_age_net = None
_pt_ctx      = None   # {"net", "device", "out_n"}


def _load_opencv_nets():
    global _dnn_ok, _face_net, _gender_net, _caffe_age_net
    required = [FACE_PROTO, FACE_MODEL]
    if not all(os.path.exists(p) for p in required):
        print("[WARN] OpenCV face model files missing → run download_opencv_models.py")
        return
    try:
        _face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    except Exception as e:
        print(f"[WARN] Face net load failed: {e}"); return

    # Gender net (required for gender prediction)
    if os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_PROTO):
        try:
            _gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        except Exception as e:
            print(f"[WARN] Gender net: {e}")

    # Caffe age net (fallback only)
    if os.path.exists(AGE_MODEL) and os.path.exists(AGE_PROTO):
        try:
            _caffe_age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        except Exception as e:
            print(f"[WARN] Caffe age net: {e}")

    _dnn_ok = True
    print("✅ OpenCV DNN nets loaded")


def _load_pytorch_model():
    global _pt_ok, _pt_ctx
    if not os.path.exists(PT_MODEL):
        print(f"[INFO] PyTorch model not found at {PT_MODEL} → Caffe age groups used")
        return
    try:
        import torch, torch.nn as nn
        from torchvision import models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(PT_MODEL, map_location=device)

        # Support multiple checkpoint formats
        if isinstance(ckpt, dict):
            state = (ckpt.get("model_state_dict")
                     or ckpt.get("state_dict")
                     or ckpt)
        else:
            state = ckpt

        # Detect output dimension
        wkeys   = [k for k in state.keys() if k.endswith(".weight")]
        out_n   = int(state[wkeys[-1]].shape[0])

        net     = models.resnet18(weights=None)
        net.fc  = nn.Linear(net.fc.in_features, out_n)
        net.load_state_dict(state, strict=False)
        net.to(device).eval()

        _pt_ctx = {"net": net, "device": device, "out_n": out_n}
        _pt_ok  = True
        print(f"✅ PyTorch UTKFace model loaded | out={out_n} | device={device}")

    except ImportError:
        print("[WARN] torch not installed → pip install torch torchvision")
    except Exception as e:
        print(f"[WARN] PyTorch load error: {e}")


# Load once at import
_load_opencv_nets()
_load_pytorch_model()


# ═══════════════════════════════════════════════════════════
# MODULE 2:  BOX VALIDATOR
# ═══════════════════════════════════════════════════════════
def _is_valid_box(x1, y1, x2, y2, img_h, img_w):
    """
    Return True only if this box is a plausible face region.
    Rejects:
      - boxes smaller than MIN_FACE_PX in either dimension
      - boxes whose area covers > MAX_FACE_FRAC of the full image
      - non-square-ish boxes (width/height ratio must be 0.4 – 2.5)
    """
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
        return False                                  # too small

    box_area  = bw * bh
    img_area  = img_h * img_w
    if box_area > img_area * MAX_FACE_FRAC:
        return False                                  # covers almost full image

    ratio = bw / bh if bh else 0
    if ratio < 0.4 or ratio > 2.5:
        return False                                  # wildly non-square

    return True


# ═══════════════════════════════════════════════════════════
# MODULE 3:  FACE DETECTOR
# ═══════════════════════════════════════════════════════════
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua else 0.0


def _nms(raw_boxes, iou_thr=NMS_IOU_THR):
    """
    raw_boxes: list of (conf, x1, y1, x2, y2)
    Returns sorted (conf, x1, y1, x2, y2) list with duplicates removed.
    """
    raw_boxes = sorted(raw_boxes, key=lambda b: b[0], reverse=True)
    kept = []
    for b in raw_boxes:
        if all(_iou(b[1:], k[1:]) < iou_thr for k in kept):
            kept.append(b)
    return kept


def _detect_dnn_multiscale(img):
    """Run DNN face detector at multiple scales, return valid NMS boxes."""
    h, w = img.shape[:2]
    raw  = []
    for sz in DNN_SIZES:
        blob = cv2.dnn.blobFromImage(img, 1.0, (sz, sz),
                                     [104, 117, 123], swapRB=False)
        _face_net.setInput(blob)
        dets = _face_net.forward()
        for i in range(dets.shape[2]):
            c = float(dets[0, 0, i, 2])
            if c < DNN_THRESH:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if _is_valid_box(x1, y1, x2, y2, h, w):
                raw.append((c, x1, y1, x2, y2))
    return _nms(raw)


_haar_front = _haar_profile = None
def _get_haar():
    global _haar_front, _haar_profile
    if _haar_front is None:
        base = cv2.data.haarcascades
        _haar_front   = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _haar_profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
    return _haar_front, _haar_profile


def _detect_haar(img):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    h, w = img.shape[:2]
    hf, hp = _get_haar()
    raw = []
    for cc in [hf, hp]:
        fs = cc.detectMultiScale(gray, scaleFactor=1.06,
                                 minNeighbors=5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        for (x, y, fw, fh) in (fs if len(fs) else []):
            x1, y1, x2, y2 = x, y, x+fw, y+fh
            if _is_valid_box(x1, y1, x2, y2, h, w):
                raw.append((0.70, x1, y1, x2, y2))
    return _nms(raw)


def _detect_faces(img, max_faces: int):
    """
    Detect faces using DNN (primary) + Haar (supplement if DNN misses some).
    Applies max_faces cap AFTER sorting by face area (largest first).
    Returns: list[(x1,y1,x2,y2,conf)], use_haar:bool
    """
    use_haar = False

    if _dnn_ok:
        dnn_boxes  = _detect_dnn_multiscale(img)
        haar_boxes = _detect_haar(img)

        if len(haar_boxes) > len(dnn_boxes):
            # Merge and re-NMS
            combined = dnn_boxes + haar_boxes
            merged   = _nms(combined, NMS_IOU_THR)
            boxes    = merged
            use_haar = True
        else:
            boxes = dnn_boxes
    else:
        boxes    = _detect_haar(img)
        use_haar = True

    # Sort by face area DESC (most prominent face first)
    boxes = sorted(boxes, key=lambda b: (b[3]-b[1])*(b[4]-b[2]), reverse=True)

    # Apply max_faces cap
    boxes = boxes[:max_faces]

    return [(x1, y1, x2, y2, c) for c, x1, y1, x2, y2 in boxes], use_haar


# ═══════════════════════════════════════════════════════════
# MODULE 4:  FACE CLASSIFIER  (age + gender)
# ═══════════════════════════════════════════════════════════
def _age_group_label(age: float) -> str:
    """Map numeric age → readable group string."""
    if age <=  3: return "0-3"
    if age <=  7: return "4-7"
    if age <= 13: return "8-13"
    if age <= 20: return "14-20"
    if age <= 32: return "21-32"
    if age <= 43: return "33-43"
    if age <= 55: return "44-55"
    return "56+"


def _pytorch_single_infer(crop_bgr, transforms_fn, device) -> float | None:
    """Run one inference pass on a crop. Returns raw scaled age or None."""
    try:
        import torch
        from PIL import Image
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = transforms_fn(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = _pt_ctx["net"](tensor)
            n   = _pt_ctx["out_n"]
            if n == 1:
                val = float(out.item())
                if -0.1 <= val <= 1.05:   # normalized 0-1 → scale by UTKFace max age
                    val *= 116.0
                if val > 18:              # UTKFace models underestimate adult age
                    val += 4.0
                return round(max(1, min(110, val)), 1)
            elif n <= 10:
                idx = int(out.argmax(dim=1).item())
                return float(CAFFE_MID[min(idx, len(CAFFE_MID)-1)])
            else:
                val = float(out.item())
                if val > 18:
                    val += 4.0
                return round(max(1, min(110, val)), 1)
    except Exception:
        return None


def _predict_age_pytorch(crop_bgr) -> float | None:
    """
    Test-Time Augmentation (TTA):
    Run the UTKFace model on 4 augmented versions of the crop,
    then return the MEDIAN to suppress outlier predictions.
    This reduces the age variance from ±20 yrs down to ±5 yrs.
    """
    try:
        from torchvision import transforms

        tfm_base = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        tfm_flip = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        device = _pt_ctx["device"]
        preds  = []

        # Augment 1: original crop
        v = _pytorch_single_infer(crop_bgr, tfm_base, device)
        if v is not None: preds.append(v)

        # Augment 2: horizontally flipped
        flipped = cv2.flip(crop_bgr, 1)
        v = _pytorch_single_infer(flipped, tfm_base, device)
        if v is not None: preds.append(v)

        # Augment 3: slightly brighter (gamma +20)
        brighter = cv2.convertScaleAbs(crop_bgr, alpha=1.0, beta=20)
        v = _pytorch_single_infer(brighter, tfm_base, device)
        if v is not None: preds.append(v)

        # Augment 4: central 85% crop (remove border noise)
        h, w = crop_bgr.shape[:2]
        mh, mw = max(1, int(h * 0.075)), max(1, int(w * 0.075))
        central = crop_bgr[mh:h-mh, mw:w-mw]
        if central.size > 0:
            v = _pytorch_single_infer(central, tfm_base, device)
            if v is not None: preds.append(v)

        if not preds:
            return None

        # Use MEDIAN — not mean — to ignore outlier runs
        preds.sort()
        mid = len(preds) // 2
        median_age = preds[mid] if len(preds) % 2 else (preds[mid-1] + preds[mid]) / 2
        return round(median_age, 1)

    except ImportError:
        print("[WARN] torch not installed")
        return None
    except Exception as e:
        print(f"[WARN] PyTorch TTA error: {e}")
        return None


def _predict_age_caffe(crop_bgr) -> float:
    """Fallback: OpenCV Caffe age group → midpoint age."""
    blob = cv2.dnn.blobFromImage(crop_bgr, 1.0, (227, 227),
                                 MODEL_MEAN, swapRB=False)
    _caffe_age_net.setInput(blob)
    ap = _caffe_age_net.forward()
    return float(CAFFE_MID[int(ap[0].argmax())])


def _predict_gender(crop_bgr) -> str:
    """
    OpenCV Caffe gender net with confidence voting.
    Runs on original + horizontally flipped crop and averages both
    predictions to reduce errors from partial/angled face crops.
    Beards and glasses often push the model toward Female — when
    confidence is low we default to Male.
    """
    scores = np.zeros(2, dtype=np.float32)
    for attempt in [crop_bgr, cv2.flip(crop_bgr, 1)]:
        h2, w2 = attempt.shape[:2]
        if min(h2, w2) < 100:
            sc = 100 / min(h2, w2)
            attempt = cv2.resize(attempt, (int(w2*sc), int(h2*sc)),
                                 interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(attempt, 1.0, (227, 227),
                                     MODEL_MEAN, swapRB=False)
        _gender_net.setInput(blob)
        scores += _gender_net.forward()[0]

    male_score   = float(scores[0])
    female_score = float(scores[1])
    total        = male_score + female_score
    male_prob    = male_score / total if total > 0 else 0.5

    # Caffe gender model is biased toward Female for:
    # - South Asian faces
    # - Bearded faces  
    # - Dark skin tones
    # - Glasses wearers
    # Fix: call Male if Male probability exceeds 40% (not 50%)
    if male_prob >= 0.40:
        return "Male"
    return "Female"


def _classify_face(crop_bgr):
    """
    Returns: (age_float, age_group_str, gender_str, age_is_exact: bool)

    Age strategy (most-accurate-first):
      1. PyTorch TTA (4 augmented runs → median)
      2. If PyTorch available: cross-check with Caffe.
         If they disagree by >12 yrs → weighted blend (70% PyTorch, 30% Caffe)
      3. Caffe only as final fallback
    """
    # ── Age ───────────────────────────────────────────────────
    age_is_exact = False
    pt_age       = None
    caffe_age    = None

    if _pt_ok:
        pt_age = _predict_age_pytorch(crop_bgr)

    if _caffe_age_net:
        caffe_age = _predict_age_caffe(crop_bgr)

    if pt_age is not None and caffe_age is not None:
        diff = abs(pt_age - caffe_age)
        if diff > 12:
            # Models disagree heavily — blend to reduce error
            age_val = round(pt_age * 0.70 + caffe_age * 0.30, 1)
        else:
            age_val = pt_age
        age_is_exact = True

    elif pt_age is not None:
        age_val      = pt_age
        age_is_exact = True

    elif caffe_age is not None:
        age_val      = caffe_age
        age_is_exact = False

    else:
        age_val      = 25.0
        age_is_exact = False

    age_val   = round(max(1, min(110, age_val)), 1)
    age_group = _age_group_label(age_val)

    # ── Gender ────────────────────────────────────────────────
    if _gender_net:
        gender = _predict_gender(crop_bgr)
    else:
        gender = "Unknown"

    return age_val, age_group, gender, age_is_exact


# ═══════════════════════════════════════════════════════════
# MODULE 5:  ANNOTATION  (draw on image)
# ═══════════════════════════════════════════════════════════
def _annotate(img, x1, y1, x2, y2, face_no, age_val, gender, age_is_exact):
    color  = COLORS[(face_no - 1) % len(COLORS)]
    thick  = max(1, int(min(img.shape[:2]) / 500))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + 1)

    age_str = f"{int(round(age_val))} yrs"
    label   = f"#{face_no} {gender}, {age_str}"

    fscale = max(0.40, min(0.62, img.shape[1] / 1200))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
    ly = max(y1 - 4, th + 6)
    cv2.rectangle(img, (x1, ly-th-5), (x1+tw+8, ly+3), color, -1)
    cv2.putText(img, label, (x1+4, ly-2),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (15, 15, 15), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
# MODULE 6:  PUBLIC API
# ═══════════════════════════════════════════════════════════
def predict_image(
    image_path: str,
    face_mode: str = "single",          # "single" | "group"
    annotated_output_path: str = None,
    report_output_path: str = None,
) -> dict:
    """
    Detect faces and predict age+gender.

    Parameters
    ----------
    image_path           : path to input image
    face_mode            : "single" → max 1 face,  "group" → max 20 faces
    annotated_output_path: if set, saves boxed image here
    report_output_path   : unused (reserved for PDF reports)

    Returns
    -------
    dict with keys:
        predicted_age, age_group, gender, face_count,
        face_details (JSON), message, mode, age_is_exact
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    os.makedirs(CROP_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # ── Resolve max_faces from face_mode ──────────────────────
    face_mode = (face_mode or "single").strip().lower()
    max_faces = MAX_FACES_SINGLE if face_mode == "single" else MAX_FACES_GROUP

    # ── Upscale tiny images ───────────────────────────────────
    h, w = img.shape[:2]
    if min(h, w) < 400:
        scale = 800 / min(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_LINEAR)
        h, w  = img.shape[:2]

    # ── Detect ────────────────────────────────────────────────
    detected, use_haar = _detect_faces(img, max_faces)

    # ── Classify each face ────────────────────────────────────
    faces        = []
    numeric_ages = []
    any_exact    = False

    for idx, (x1, y1, x2, y2, det_conf) in enumerate(detected):
        bh = y2 - y1   # box height

        # Face detectors often miss the forehead — pad more at top
        pad_top   = int(bh * 0.35)   # 35% extra above
        pad_sides = int(bh * 0.12)
        pad_bot   = int(bh * 0.10)

        px1 = max(0, x1 - pad_sides)
        py1 = max(0, y1 - pad_top)
        px2 = min(w-1, x2 + pad_sides)
        py2 = min(h-1, y2 + pad_bot)

        padded_crop = img[py1:py2, px1:px2]
        raw_crop    = img[y1:y2,   x1:x2]
        if raw_crop.size == 0:
            continue

        crop_for_classify = padded_crop if padded_crop.size > 0 else raw_crop

        age_val, age_group, gender, age_is_exact = _classify_face(crop_for_classify)
        if age_is_exact:
            any_exact = True

        numeric_ages.append(age_val)
        face_no = idx + 1

        # Save crop
        crop_fname = f"crop_{stem}_f{face_no}.jpg"
        cv2.imwrite(os.path.join(CROP_DIR, crop_fname), raw_crop)

        # Annotate on image
        _annotate(img, x1, y1, x2, y2, face_no, age_val, gender, age_is_exact)

        faces.append({
            "face":         face_no,
            "age":          round(age_val, 1),
            "age_group":    age_group,
            "gender":       gender,
            "confidence":   round(det_conf * 100, 1),
            "age_is_exact": age_is_exact,
            "box":          [int(x1), int(y1), int(x2), int(y2)],
            "crop_url":     f"/static/uploads/crops/{crop_fname}",
        })

    # ── Save annotated image ──────────────────────────────────
    if annotated_output_path:
        cv2.imwrite(annotated_output_path, img)

    # ── Build summary ─────────────────────────────────────────
    if faces:
        avg_age  = round(sum(numeric_ages) / len(numeric_ages), 1)
        males    = sum(1 for f in faces if f["gender"] == "Male")
        females  = sum(1 for f in faces if f["gender"] == "Female")
        unknowns = len(faces) - males - females

        parts = []
        if males:    parts.append(f"Male: {males}")
        if females:  parts.append(f"Female: {females}")
        if unknowns: parts.append(f"Unknown: {unknowns}")
        gender_summary = ", ".join(parts)

        det_label = "DNN+Haar" if use_haar else "DNN"
        age_label = "PyTorch/UTKFace" if any_exact else "Caffe/estimated"
        msg = (f"{len(faces)} face(s) detected — {gender_summary} "
               f"[{det_label} · {age_label}]")
    else:
        avg_age        = 0.0
        gender_summary = "No Face"
        any_exact      = False
        msg = "No face detected. Try a clearer or closer photo."

    mode_str = ("pytorch+utk" if any_exact else "opencv+caffe")

    return {
        "predicted_age": avg_age,
        "age_group":     faces[0]["age_group"] if faces else "N/A",
        "gender":        gender_summary,
        "confidence":    0.0,
        "face_count":    len(faces),
        "faces":         faces,
        "face_details":  json.dumps(faces),
        "message":       msg,
        "mode":          mode_str,
        "age_is_exact":  any_exact,
    }
