import os
import re
import secrets
import sqlite3
import threading
from functools import wraps

from flask import g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "users.db")
TOKEN_MAX_AGE = 7 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")
PASSWORD_MIN, PASSWORD_MAX = 6, 128


def _secret():
    env = os.environ.get("TANK_SECRET")
    if env:
        return env
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "secret.key")
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


_serializer = URLSafeTimedSerializer(_secret(), salt="tank-arena-auth")
_db_lock = threading.Lock()


def _db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " username TEXT UNIQUE NOT NULL COLLATE NOCASE,"
        " pw_hash TEXT NOT NULL,"
        " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def _session(user):
    return {"token": _serializer.dumps(user), "user": user}


def verify_token(token):
    if not isinstance(token, str) or not token:
        return None
    try:
        return _serializer.loads(token, max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def current_user():
    header = request.headers.get("Authorization", "")
    return verify_token(header[7:]) if header.startswith("Bearer ") else None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Please log in again."}), 401
        g.user = user
        return fn(*args, **kwargs)
    return wrapper


def _credentials():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    return username, password


def register_auth_routes(app):
    @app.post("/api/auth/register")
    def register():
        username, password = _credentials()
        if not USERNAME_RE.match(username):
            return jsonify({"error": "Username must be 3-20 letters, digits or underscores."}), 400
        if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
            return jsonify({"error": f"Password must be {PASSWORD_MIN}-{PASSWORD_MAX} characters."}), 400
        try:
            with _db_lock, _db() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, pw_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                uid = cur.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({"error": "That username is taken."}), 409
        return jsonify(_session({"id": f"u-{uid}", "name": username, "guest": False}))

    @app.post("/api/auth/login")
    def login():
        username, password = _credentials()
        with _db_lock, _db() as conn:
            row = conn.execute(
                "SELECT id, username, pw_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None or not check_password_hash(row[2], password):
            return jsonify({"error": "Invalid username or password."}), 401
        return jsonify(_session({"id": f"u-{row[0]}", "name": row[1], "guest": False}))

    @app.post("/api/auth/guest")
    def guest():
        return jsonify(_session({
            "id": f"g-{secrets.token_hex(8)}",
            "name": f"Guest-{secrets.randbelow(9000) + 1000}",
            "guest": True,
        }))

    @app.get("/api/auth/me")
    @login_required
    def me():
        return jsonify({"user": g.user})
