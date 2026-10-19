import pytest
from ekip_config.settings import Settings
from pydantic import ValidationError

SETTINGS_ENV = (
    "APP_ENV",
    "LOG_LEVEL",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "OPENSEARCH_URL",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
    "OIDC_JWKS_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


def make() -> Settings:
    return Settings(_env_file=None)


def test_defaults_load_with_empty_environment() -> None:
    s = make()
    assert s.app_env == "local"
    assert s.log_level == "INFO"
    assert str(s.llm_base_url) == "https://api.groq.com/openai/v1"


def test_global_openai_base_url_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil.example/v1")
    assert str(make().llm_base_url) == "https://api.groq.com/openai/v1"
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    assert str(make().llm_base_url) == "https://llm.example/v1"


def test_secrets_are_masked_but_retrievable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "sk-test-123")
    s = make()
    assert "sk-test-123" not in repr(s)
    assert "sk-test-123" not in str(s)
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "sk-test-123"


def test_invalid_log_level_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
    with pytest.raises(ValidationError):
        make()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db"),
        ("REDIS_URL", "redis://127.0.0.1:6379/0"),
        ("OPENSEARCH_URL", "http://localhost:9200"),
    ],
)
def test_production_with_local_infra_fails(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.internal:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/0")
    monkeypatch.setenv("OPENSEARCH_URL", "https://search.internal:9200")
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com/realms/ekip")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://idp.example.com/realms/ekip/certs")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match="localhost in production"):
        make()


def test_production_with_remote_infra_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.internal:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/0")
    monkeypatch.setenv("OPENSEARCH_URL", "https://search.internal:9200")
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com/realms/ekip")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://idp.example.com/realms/ekip/certs")
    assert make().app_env == "production"


def test_settings_are_frozen() -> None:
    s = make()
    with pytest.raises(ValidationError):
        s.log_level = "DEBUG"  # type: ignore[misc]


def test_oidc_defaults_are_local_dev_values() -> None:
    s = make()
    assert str(s.oidc_issuer) == "http://localhost:8081/realms/ekip"
    assert s.oidc_audience == "ekip-api"


def _production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.internal:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/0")
    monkeypatch.setenv("OPENSEARCH_URL", "https://search.internal:9200")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OIDC_ISSUER", "http://idp.example.com/realms/ekip", "OIDC_ISSUER must use https"),
        ("OIDC_JWKS_URL", "http://idp.example.com/certs", "OIDC_JWKS_URL must use https"),
        ("OIDC_ISSUER", "https://localhost/realms/ekip", "OIDC_ISSUER points to localhost"),
        ("OIDC_JWKS_URL", "https://127.0.0.1/certs", "OIDC_JWKS_URL points to localhost"),
    ],
)
def test_production_requires_secure_non_local_oidc(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    _production_env(monkeypatch)
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com/realms/ekip")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://idp.example.com/certs")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=message):
        make()


def test_production_with_default_oidc_urls_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _production_env(monkeypatch)
    with pytest.raises(ValidationError, match="OIDC_ISSUER"):
        make()
