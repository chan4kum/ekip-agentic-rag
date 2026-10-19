from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, SecretStr, model_validator
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

    # Embeddings: a different provider/model lifecycle from the chat LLM (see ADR on model_version).
    embedding_provider: Literal["gemini", "hashing"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)

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
    kafka_bootstrap_servers: str = "localhost:19092"  # comma-separated host:port list
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "ekip-ingestion"
    kafka_topic_prefix: str = Field(default="ekip", pattern=r"^[a-z][a-z0-9_-]{0,31}$")

    # Hard per-request deadline (non-streaming routes). Keep it below the proxy's response-header
    # timeout (60 s in Traefik) so callers get our problem+json 504, not an opaque proxy error.
    request_timeout_s: float = Field(default=30.0, gt=0, le=55.0)

    # Largest accepted request body. Traefik enforces the same 1 MiB for ordinary routes; the app
    # enforces it itself for streams, which bypass the edge's buffering middleware.
    max_request_body_bytes: int = Field(default=1_048_576, ge=1024)

    # Streaming chat (SSE): own overall deadline (the 30 s request deadline would cut answers off),
    # heartbeats so idle proxies keep the connection, and a typing delay for the Phase-2 stub only.
    chat_timeout_s: float = Field(default=120.0, gt=0, le=600.0)
    chat_keepalive_s: float = Field(default=15.0, gt=0, le=60.0)
    chat_stub_token_delay_s: float = Field(default=0.0, ge=0, le=2.0)

    # Rate limiting (token buckets in Redis). Burst = bucket size; per_minute = refill speed.
    # The edge (Traefik) already limits per IP; these limit per authenticated user and tenant.
    rate_limit_user_burst: int = Field(default=20, ge=1)
    rate_limit_user_per_minute: int = Field(default=60, ge=1)
    rate_limit_tenant_burst: int = Field(default=200, ge=1)
    rate_limit_tenant_per_minute: int = Field(default=600, ge=1)
    rate_limit_fail_open: bool = False  # Redis down: False -> 503 (safe default), True -> allow

    # OIDC (Keycloak locally). The issuer is what tokens must carry in `iss` (the browser-facing
    # URL); the JWKS URL is where WE fetch keys (may be an internal address). We never derive one
    # from the other, and never take either from a token (that would let an attacker pick keys).
    oidc_issuer: AnyHttpUrl = AnyHttpUrl("http://localhost:8081/realms/ekip")
    oidc_audience: str = "ekip-api"
    oidc_jwks_url: AnyHttpUrl = AnyHttpUrl(
        "http://localhost:8081/realms/ekip/protocol/openid-connect/certs"
    )

    @model_validator(mode="after")
    def _secure_oidc_in_production(self) -> Self:
        if self.app_env == "production":
            for name, url in (
                ("OIDC_ISSUER", self.oidc_issuer),
                ("OIDC_JWKS_URL", self.oidc_jwks_url),
            ):
                if url.scheme != "https":
                    raise ValueError(f"{name} must use https in production")
                if (url.host or "") in {"localhost", "127.0.0.1"}:
                    raise ValueError(f"{name} points to localhost in production")
        return self

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
