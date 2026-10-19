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
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match="localhost in production"):
        make()


def test_production_with_remote_infra_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.internal:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/0")
    monkeypatch.setenv("OPENSEARCH_URL", "https://search.internal:9200")
    assert make().app_env == "production"


def test_settings_are_frozen() -> None:
    s = make()
    with pytest.raises(ValidationError):
        s.log_level = "DEBUG"  # type: ignore[misc]
