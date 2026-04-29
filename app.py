"""
app.py — AI FaceAge Flask Application  (v9 — fully fixed)
==========================================================
Fixes applied:
  1. Cloudinary import is now conditional (no crash if not installed)
  2. Postgres migration added (_postgres_migrate) — adds missing columns
     to existing Postgres tables without dropping data
  3. SQLite migration retained (_sqlite_migrate)
  4. DB_IS_POSTGRES imported from config
  5. All routes consistent with v8 template set
"""

import os, json
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file)
from werkzeug.utils import secure_filename
from sqlalchemy import text

from config import (
    SECRET_KEY, DEBUG,
    SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS,
    SQLALCHEMY_ENGINE_OPTIONS,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, DB_IS_POSTGRES,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET, USE_CLOUDINARY,
)
from models import db, User, Prediction, AdminLog
from predict_fixed import predict_image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Cloudinary — optional, graceful import ────────────────
cloudinary = None
cloudinary_uploader = None
if USE_CLOUDINARY:
    try:
        import cloudinary as _cl
        import cloudinary.uploader as _cl_up
        cloudinary = _cl
        cloudinary_uploader = _cl_up
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET,
            secure=True,
        )
        print("✅ Cloudinary configured")
    except ImportError:
        print("[WARN] cloudinary package not installed — USE_CLOUDINARY forced off")
        USE_CLOUDINARY = False

# ── Flask app ──────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]                     = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"]        = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config["SQLALCHEMY_ENGINE_OPTIONS"]      = SQLALCHEMY_ENGINE_OPTIONS
app.config["UPLOAD_FOLDER"]                  = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "crops"), exist_ok=True)

db.init_app(app)


# ══════════════════════════════════════════════════════════
#  DB initialisation — works for both SQLite & Postgres
# ══════════════════════════════════════════════════════════

# Columns added in later versions — may be absent in old DBs
_NEW_COLS = {
    "face_details": "TEXT",
    "report_url":   "VARCHAR(500)",
    "message":      "VARCHAR(400)",
    "mode":         "VARCHAR(40)",
}


def init_db():
    """Create tables if they don't exist; migrate missing columns."""
    try:
        db.create_all()
        print("✅ DB tables ready")
    except Exception as e:
        print(f"❌ db.create_all() error: {e}")
        raise

    if DB_IS_POSTGRES:
        _postgres_migrate()
    else:
        _sqlite_migrate()

    _create_admin()


def _postgres_migrate():
    """
    Add missing columns to the 'predictions' table in Postgres.
    Uses information_schema — safe no-op if column already exists.
    """
    try:
        with db.engine.connect() as conn:
            for col, ctype in _NEW_COLS.items():
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name='predictions' AND column_name=:col"
                    ),
                    {"col": col},
                ).fetchone()
                if not exists:
                    conn.execute(
                        text(f"ALTER TABLE predictions ADD COLUMN {col} {ctype}")
                    )
                    print(f"  [migrate] Postgres: added column '{col}'")
            conn.commit()
    except Exception as e:
        print(f"[WARN] Postgres migrate: {e}")


def _sqlite_migrate():
    """Add new columns to old SQLite databases (safe no-op if already exist)."""
    try:
        with db.engine.connect() as conn:
            rows     = conn.exec_driver_sql("PRAGMA table_info(predictions)").fetchall()
            existing = {r[1] for r in rows}
            for col, ctype in _NEW_COLS.items():
                if col not in existing:
                    conn.exec_driver_sql(
                        f"ALTER TABLE predictions ADD COLUMN {col} {ctype}"
                    )
                    print(f"  [migrate] SQLite: added column '{col}'")
            conn.commit()
    except Exception as e:
        print(f"[WARN] SQLite migrate: {e}")


def _create_admin():
    try:
        if not User.query.filter_by(email="admin@gmail.com").first():
            a = User(name="Admin", email="admin@gmail.com", role="admin")
            a.set_password("admin123")
            db.session.add(a)
            db.session.commit()
            print("✅ Default admin created: admin@gmail.com / admin123")
    except Exception as e:
        db.session.rollback()
        print(f"[WARN] Admin create: {e}")


# ── Auth helpers ───────────────────────────────────────────
def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    try:
        return db.session.get(User, uid)
    except Exception:
        return None


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if not u:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        if u.blocked:
            session.clear()
            flash("Your account is blocked.", "error")
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if not u or u.role != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


@app.context_processor
def inject_globals():
    return {
        "auth_user":      current_user(),
        "use_cloudinary": USE_CLOUDINARY,
        "db_backend":     "Supabase PostgreSQL" if DB_IS_POSTGRES else "SQLite (local)",
    }


# ── Health check (Render uses this) ───────────────────────
@app.route("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "db": db_ok, "cloudinary": USE_CLOUDINARY}, 200


# ── Public pages ───────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


@app.route("/about")
def about():
    team = [
        {"name": "Md Raisul Islam",        "id": "231902037"},
        {"name": "Md Mahabub Hasan Mahin", "id": "231902056"},
        {"name": "Chinmoy Debnath",        "id": "231902029"},
    ]
    return render_template("about.html", team=team)


@app.route("/index.html")
@app.route("/index")
def index_launcher():
    p = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(p):
        return send_file(p)
    return redirect(url_for("home"))


# ── Auth ───────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name     = request.form.get("name",     "").strip()
        email    = request.form.get("email",    "").lower().strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("signup"))

        try:
            if User.query.filter_by(email=email).first():
                flash("Email already registered. Please login.", "error")
                return redirect(url_for("signup"))
            u = User(name=name, email=email)
            u.set_password(password)
            db.session.add(u)
            db.session.commit()
            flash(f"Account created! Welcome, {name}. Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash(f"Registration error: {e}", "error")
            return redirect(url_for("signup"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email",    "").lower().strip()
        password = request.form.get("password", "")

        try:
            u = User.query.filter_by(email=email).first()
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return redirect(url_for("login"))

        if not u or not u.check_password(password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))
        if u.blocked:
            flash("Your account is blocked. Contact admin.", "error")
            return redirect(url_for("login"))

        session.permanent = True
        session["user_id"] = u.id
        flash(f"Welcome back, {u.name}! 👋", "success")
        return redirect(
            url_for("admin_dashboard") if u.role == "admin" else url_for("dashboard")
        )

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


# ── User pages ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    u     = current_user()
    preds = u.predictions.order_by(Prediction.created_at.desc()).all()
    return render_template("dashboard.html", predictions=preds)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No image selected.", "error")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Only JPG, JPEG, PNG, WEBP allowed.", "error")
            return redirect(url_for("upload"))

        filename    = secure_filename(file.filename)
        unique_name = datetime.now().strftime("%Y%m%d%H%M%S_") + filename
        save_path   = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(save_path)

        annotated_name = "boxed_" + unique_name
        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)

        face_mode = request.form.get("face_mode", "single").strip().lower()
        if face_mode not in ("single", "group"):
            face_mode = "single"

        try:
            result = predict_image(
                save_path,
                face_mode=face_mode,
                annotated_output_path=annotated_path,
            )
        except FileNotFoundError as e:
            flash(str(e), "error")
            return redirect(url_for("upload"))
        except Exception as e:
            flash(f"AI error: {e}", "error")
            return redirect(url_for("upload"))

        display_name = annotated_name if os.path.exists(annotated_path) else unique_name
        display_path = annotated_path if os.path.exists(annotated_path) else save_path

        # ── Cloudinary upload (optional) ──────────────────────
        if USE_CLOUDINARY and cloudinary_uploader:
            try:
                cl = cloudinary_uploader.upload(
                    display_path,
                    folder="age_ai_faceage",
                    public_id=display_name.rsplit(".", 1)[0],
                    overwrite=True,
                    resource_type="image",
                )
                image_url = cl["secure_url"]

                # Upload face crops
                face_details = json.loads(result.get("face_details", "[]"))
                for face in face_details:
                    crop_local = os.path.join(
                        BASE_DIR, face.get("crop_url", "").lstrip("/")
                    )
                    if os.path.exists(crop_local):
                        try:
                            c2 = cloudinary_uploader.upload(
                                crop_local,
                                folder="age_ai_faceage/crops",
                                public_id=os.path.basename(crop_local).rsplit(".", 1)[0],
                                overwrite=True,
                            )
                            face["crop_url"] = c2["secure_url"]
                            os.remove(crop_local)
                        except Exception:
                            pass
                result["face_details"] = json.dumps(face_details)

                for p in [save_path, display_path]:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            except Exception as e:
                flash(f"Cloudinary upload failed: {e}. Using local storage.", "warning")
                image_url = url_for("static", filename=f"uploads/{display_name}")
        else:
            image_url = url_for("static", filename=f"uploads/{display_name}")
            if os.path.exists(save_path) and save_path != display_path:
                try:
                    os.remove(save_path)
                except Exception:
                    pass

        # ── Save to DB ────────────────────────────────────────
        try:
            pred = Prediction(
                user_id       = current_user().id,
                image_url     = image_url,
                file_name     = display_name,
                predicted_age = result.get("predicted_age", 0.0),
                age_group     = result.get("age_group",     "N/A"),
                confidence    = 0.0,
                gender        = result.get("gender",        ""),
                emotion       = result.get("emotion",       ""),
                face_count    = result.get("face_count",     0),
                face_details  = result.get("face_details",  "[]"),
                message       = result.get("message",       ""),
                mode          = result.get("mode",          "opencv"),
            )
            db.session.add(pred)
            db.session.commit()
            return redirect(url_for("result", prediction_id=pred.id))
        except Exception as e:
            db.session.rollback()
            flash(f"DB save error: {e}", "error")
            return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    u    = current_user()
    if u.role != "admin" and pred.user_id != u.id:
        flash("Access denied.", "error")
        return redirect(url_for("dashboard"))
    try:
        face_details = json.loads(pred.face_details or "[]")
    except Exception:
        face_details = []
    return render_template("result.html", pred=pred, face_details=face_details)


@app.route("/delete-result/<int:prediction_id>", methods=["POST"])
@login_required
def delete_result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    u    = current_user()
    if u.role != "admin" and pred.user_id != u.id:
        return redirect(url_for("dashboard"))

    if USE_CLOUDINARY and cloudinary_uploader and "cloudinary.com" in (pred.image_url or ""):
        try:
            pid = "age_ai_faceage/" + pred.file_name.rsplit(".", 1)[0]
            cloudinary_uploader.destroy(pid)
        except Exception:
            pass
        try:
            for f in json.loads(pred.face_details or "[]"):
                if "cloudinary.com" in f.get("crop_url", ""):
                    c_pid = f["crop_url"].split("/upload/")[1].rsplit(".", 1)[0]
                    cloudinary_uploader.destroy(c_pid)
        except Exception:
            pass
    else:
        try:
            p = os.path.join(UPLOAD_FOLDER, pred.file_name or "")
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    try:
        db.session.delete(pred)
        db.session.commit()
        flash("Result deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Delete error: {e}", "error")

    return redirect(
        url_for("dashboard") if u.role != "admin" else url_for("admin_dashboard")
    )


@app.route("/download-result/<int:prediction_id>")
@login_required
def download_result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    if "cloudinary.com" in (pred.image_url or ""):
        return redirect(pred.image_url)
    p = os.path.join(UPLOAD_FOLDER, pred.file_name or "")
    if not os.path.exists(p):
        flash("File not found.", "error")
        return redirect(url_for("result", prediction_id=prediction_id))
    return send_file(p, as_attachment=True)


@app.route("/profile")
@login_required
def profile():
    u     = current_user()
    total = u.predictions.count()
    return render_template("profile.html", total_predictions=total)


# ── Admin ──────────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_dashboard():
    users        = User.query.order_by(User.created_at.desc()).all()
    preds        = Prediction.query.order_by(Prediction.created_at.desc()).all()
    logs         = AdminLog.query.order_by(AdminLog.created_at.desc()).limit(30).all()
    active_users = sum(1 for u in users if not u.blocked)
    blocked_u    = sum(1 for u in users if u.blocked)

    storage = 0
    for root_dir, _, files in os.walk(UPLOAD_FOLDER):
        for f in files:
            try:
                storage += os.path.getsize(os.path.join(root_dir, f))
            except Exception:
                pass

    return render_template(
        "admin.html",
        users=users, predictions=preds, logs=logs,
        total_users=len(users), total_predictions=len(preds),
        storage_mb=round(storage / (1024 * 1024), 2),
        active_users=active_users, blocked_users=blocked_u,
        recent_uploads=preds[:8], use_cloudinary=USE_CLOUDINARY,
        db_backend="Supabase PostgreSQL" if DB_IS_POSTGRES else "SQLite (local)",
    )


@app.route("/admin/block/<int:user_id>", methods=["POST"])
@admin_required
def block_user(user_id):
    u = User.query.get_or_404(user_id)
    if u.role == "admin":
        flash("Cannot block another admin.", "error")
        return redirect(url_for("admin_dashboard"))
    u.blocked = not u.blocked
    try:
        log = AdminLog(
            admin_id=current_user().id,
            action=f"{'Blocked' if u.blocked else 'Unblocked'}: {u.email}",
            target_user_id=u.id,
        )
        db.session.add(log)
        db.session.commit()
        flash(f"User {'blocked' if u.blocked else 'unblocked'}.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    u = User.query.get_or_404(user_id)
    if u.role == "admin":
        flash("Cannot delete admin.", "error")
        return redirect(url_for("admin_dashboard"))
    try:
        log = AdminLog(
            admin_id=current_user().id,
            action=f"Deleted user: {u.email}",
            target_user_id=u.id,
        )
        db.session.add(log)
        db.session.delete(u)
        db.session.commit()
        flash("User deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}", "error")
    return redirect(url_for("admin_dashboard"))


# ── Startup ────────────────────────────────────────────────
with app.app_context():
    init_db()

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  AI FaceAge  |  Debug={DEBUG}")
    print(f"  DB      : {'Supabase PostgreSQL' if DB_IS_POSTGRES else 'SQLite (local)'}")
    print(f"  Cloudinary: {USE_CLOUDINARY}")
    print(f"{'='*50}\n")
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
