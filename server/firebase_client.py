"""
firebase_client.py
===================
Wrapper tipis untuk Firebase Admin SDK.
Inisialisasi, listen RTDB, dan tulis hasil prediksi.
"""

import firebase_admin
from firebase_admin import credentials, db
from config import FIREBASE_DATABASE_URL, FIREBASE_CREDENTIALS_PATH


def init():
    """Inisialisasi Firebase Admin SDK. Panggil sekali di startup."""
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
    print(f"[Firebase] Terhubung → {FIREBASE_DATABASE_URL}")


def listen(path: str, callback) -> db.Reference:
    """
    Pasang listener realtime ke node RTDB.
    callback(event) dipanggil setiap data di path berubah.

    Event attributes:
      event.data       — dict baru / None kalau node dihapus
      event.path       — relative path yang berubah
      event.event_type — 'put' | 'patch'
    """
    ref = db.reference(path)
    ref.listen(callback)
    return ref


def write(path: str, data: dict):
    """Timpa seluruh node dengan data baru."""
    db.reference(path).set(data)


def push(path: str, data: dict) -> str:
    """
    Tambahkan entry baru ke history node (Firebase auto-generate key).
    Return key yang baru dibuat.
    """
    return db.reference(path).push(data).key