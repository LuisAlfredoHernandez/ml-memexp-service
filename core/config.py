import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ML Service - Meme Fábrica"
    # Ruta relativa al archivo main.py
    MODEL_PATH: str = os.path.join("model_artifacts", "random_forest_v1.pkl")

settings = Settings()