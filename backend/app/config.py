"""
app/config.py
Application configuration using pydantic-settings.

Values can be overridden via environment variables or a .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data paths ---
    data_dir: str = Field(default="data", description="Root directory containing the data folders")
    employees_output_path: str = Field(default="data/hr/employees.csv", description="Path to write the processed employee CSV")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model name for encoding chunks")
    qdrant_collection: str = Field(default="finsolve_docs", description="Qdrant collection name")
    qdrant_batch_size: int = Field(default=100, description="Number of points per upsert batch")

    # --- Access role classification ---
    # Maps raw CSV department names → semantic routing bucket.
    # Departments absent from this map are skipped for Full-Time+Active employees.
    dept_bucket: dict[str, str] = Field(
        default={
            "Finance":    "Finance",
            "Technology": "Engineering",
            "Data":       "Engineering",
            "Marketing":  "Marketing",
            "HR":         "HR",
        },
        description="Department → access role bucket mapping",
    )

    # --- Chunk access roles ---
    # Maps data folder name → the specific access role for documents in that folder.
    folder_access_role: dict[str, str] = Field(
        default={
            "general":     "General",
            "hr":          "HR",
            "finance":     "Finance",
            "engineering": "Engineering",
            "marketing":   "Marketing",
        },
        description="Data folder name → access role for chunked documents",
    )
    # These roles are always included in every chunk's access_roles list.
    chunk_common_roles: list[str] = Field(
        default=["C-Level"],
        description="Access roles assigned to every document chunk so executives can view all content",
    )
    # These roles are always granted to a user when querying.
    user_common_roles: list[str] = Field(
        default=["General"],
        description="Access roles granted to every user by default to view general documents",
    )

    # --- LLM / external service keys (loaded from .env) ---
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name for chat completions")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (leave empty for local)")
    use_local_qdrant: bool = Field(
        default=True,
        description="Use qdrant-client's built-in in-memory store (no server needed). Set to False to connect to a real Qdrant server or Qdrant Cloud."
    )


# Singleton — import this everywhere instead of instantiating Settings() yourself
settings = Settings()
