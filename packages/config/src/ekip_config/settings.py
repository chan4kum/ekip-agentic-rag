from typing import Literal, Self

from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown env vars (e.g. global OPENAI_BASE_URL) are ignored
        frozen=True,  # immutable after load
    )

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Tracing exporter: "none" = spans exist (trace_id in logs) but go nowhere;
    # "console" = one JSON line per span on stdout. (Named TRACING_*, not OTEL_*, to avoid
    # colliding with the OpenTelemetry SDK's own environment variables.)
    tracing_exporter: Literal["none", "console"] = "none"

    # LLM: any OpenAI-compatible endpoint
    llm_provider: str = "groq"
    llm_base_url: AnyHttpUrl = AnyHttpUrl("https://api.groq.com/openai/v1")
    llm_model: str = ""
    llm_api_key: SecretStr | None = None

    # Provider keys: SecretStr never prints its value
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # Infrastructure
    database_url: PostgresDsn = PostgresDsn("postgresql+asyncpg://ekip:ekip@localhost:5432/ekip")
    redis_url: RedisDsn = RedisDsn("redis://localhost:6379/0")
    opensearch_url: AnyHttpUrl = AnyHttpUrl("http://localhost:9200")

    @model_validator(mode="after")
    def _no_local_infra_in_production(self) -> Self:
        if self.app_env == "production":
            # PostgresDsn can hold several hosts (HA clusters), so it needs .hosts()
            hosts = {h["host"] for h in self.database_url.hosts() if h["host"]}
            hosts |= {self.redis_url.host or "", self.opensearch_url.host or ""}
            if hosts & {"localhost", "127.0.0.1"}:
                raise ValueError("infrastructure URL points to localhost in production")
        return self


def load_settings() -> Settings:
    """Called ONLY from a composition root (e.g. the API app factory)."""
    return Settings()
