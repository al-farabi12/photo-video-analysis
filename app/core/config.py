import os

class Settings:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}
    MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet-lite4-11.onnx")
    LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_map.txt")

settings = Settings()
