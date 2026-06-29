import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ML Service - Meme Fábrica"
    MODEL_PATH: str = os.path.join("model_artifacts", "random_forest_v1.pkl")
    DATABASE_URL: str = "postgresql://memexp_user:memexp_password@db:5432/memexp_db"

    class ConfigDict:
        env_file = ".env"
        extra = "ignore"

settings = Settings()