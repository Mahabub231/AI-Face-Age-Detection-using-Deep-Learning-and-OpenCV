"""
models.py — SQLAlchemy DB models
Works with both SQLite (local) and Supabase PostgreSQL (online)
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer,     primary_key=True)
    name          = db.Column(db.String(120),  nullable=False)
    email         = db.Column(db.String(150),  unique=True, nullable=False)
    password_hash = db.Column(db.String(256),  nullable=False)
    role          = db.Column(db.String(20),   default="user")   # "user" | "admin"
    blocked       = db.Column(db.Boolean,      default=False)
    created_at    = db.Column(db.DateTime,     default=datetime.utcnow)

    predictions = db.relationship(
        "Prediction", backref="user",
        cascade="all, delete-orphan", lazy="dynamic"
    )

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.email}>"


class Prediction(db.Model):
    __tablename__ = "predictions"

    id            = db.Column(db.Integer,      primary_key=True)
    user_id       = db.Column(db.Integer,      db.ForeignKey("users.id"), nullable=False)
    image_url     = db.Column(db.String(500),  nullable=False)
    file_name     = db.Column(db.String(255),  nullable=True)
    predicted_age = db.Column(db.Float,        nullable=False, default=0.0)
    age_group     = db.Column(db.String(40),   nullable=False, default="")
    confidence    = db.Column(db.Float,        nullable=False, default=0.0)
    gender        = db.Column(db.String(60),   nullable=False, default="")
    emotion       = db.Column(db.String(40),   default="")
    face_count    = db.Column(db.Integer,      default=0)
    face_details  = db.Column(db.Text,         nullable=True)   # JSON list
    report_url    = db.Column(db.String(500),  nullable=True)
    message       = db.Column(db.String(400),  nullable=True)
    mode          = db.Column(db.String(40),   nullable=True)
    created_at    = db.Column(db.DateTime,     default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.id} user={self.user_id}>"


class AdminLog(db.Model):
    __tablename__ = "admin_logs"

    id             = db.Column(db.Integer,    primary_key=True)
    admin_id       = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    action         = db.Column(db.String(300), nullable=False)
    target_user_id = db.Column(db.Integer,    nullable=True)
    created_at     = db.Column(db.DateTime,   default=datetime.utcnow)

    def __repr__(self):
        return f"<AdminLog {self.id}>"
