import os
from dotenv import load_dotenv

load_dotenv()

# Auto-configure local tiktoken cache if available
_tiktoken_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tiktoken_cache"))
if os.path.exists(_tiktoken_dir):
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", _tiktoken_dir)

# Disable ChromaDB telemetry in offline / containerized environments
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")


class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")

    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))

    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.0"))
    RETRY_MAX_WAIT = float(os.getenv("RETRY_MAX_WAIT", "10.0"))

    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # Cache settings
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    # Fallback provider (used when primary fails)
    FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "")  # e.g. "ollama"
    FALLBACK_VLLM_BASE_URL = os.getenv("FALLBACK_VLLM_BASE_URL", "http://localhost:11434/v1")
    FALLBACK_VLLM_MODEL = os.getenv("FALLBACK_VLLM_MODEL", "llama3.2:3b")


settings = Settings()