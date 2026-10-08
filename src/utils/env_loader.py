import os
from pathlib import Path
from dotenv import load_dotenv


def load_project_env() -> Path:
    """Load environment variables from the project's .env file."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    return env_path


def get_openai_api_key() -> str | None:
    load_project_env()
    return os.getenv("OPENAI_API_KEY")
