"""OMEGA MODEL 2: System Configuration and Environment Settings.

Manages application parameters, API keys, Model 1 integration paths,
safety boundaries, and database locations.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
MODEL1_DIR = WORKSPACE_DIR / "model 1"

# Model 1 Paths
MODEL1_CHECKPOINT = str(MODEL1_DIR / "checkpoints" / "omega_gpt1_best.pth")
MODEL1_TOKENIZER_DIR = str(MODEL1_DIR / "tokenizer_saved")

# Database & Storage
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "omega2.db"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    # Identity
    APP_NAME: str = "OMEGA MODEL 2"
    APP_VERSION: str = "2.0.0-PROD"
    DESCRIPTION: str = "Advanced Modular AI Assistant Architecture"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Model 1 (Omega GPT-1) Integration
    MODEL1_ENABLED: bool = True
    MODEL1_CHECKPOINT: str = str(MODEL1_DIR / "checkpoints" / "omega_gpt1_best.pth")
    MODEL1_TOKENIZER_DIR: str = str(MODEL1_DIR / "tokenizer_saved")

    # External LLM Providers (Read from environment variables)
    DEFAULT_PROVIDER: str = os.getenv("OMEGA_DEFAULT_PROVIDER", "omega_gpt1")
    DEFAULT_MODEL: str = os.getenv("OMEGA_DEFAULT_MODEL", "omega-gpt1-small")

    # API Keys (Never hardcoded)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Sandbox Security
    SANDBOX_TIMEOUT_SECONDS: float = 5.0
    SANDBOX_MAX_MEMORY_MB: int = 128
    ALLOW_ARBITRARY_SHELL: bool = False

    # Memory & Retention
    SHORT_TERM_MEMORY_LIMIT: int = 20
    LONG_TERM_MEMORY_SCORE_THRESHOLD: float = 0.5


# Load settings singleton
settings = Settings()
