import os
from pathlib import Path

class Config:
    """Application configuration"""
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"
    MODELS_DIR = BASE_DIR / "static" / "models"
    
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Create directories if they don't exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)