import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from ekip_config.settings import Settings, load_settings
from ekip_contracts.health import DependencyProbe
from ekip_contracts.retrieval import Retriever
from ekip_observability.logging import configure_logging, get_logger
from ekip_observability.tracing import build_tracer_provider, scrub_url_attributes
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from ekip_api.adapters.demo_data import DEMO_CHUNKS
from ekip_api.adapters.in_memory_retriever import InMemoryRetriever
from ekip_api.adapters.probes import build_default_probes
from ekip_api.middleware import RequestContextMiddleware
from ekip_api.routes import health, ready, search
from ekip_api.services.readiness import ReadinessService


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    probes: Sequence[DependencyProbe] | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Composition root: the only place that wires concrete dependencies."""
    settings = settings or load_settings()
    owns_provider = tracer_provider is None  # only shut down a provider we built ourselves
    provider = tracer_provider or build_tracer_provider(
        service="ekip-api", environment=settings.app_env, exporter=settings.tracing_exporter
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            settings.log_level,
            # pretty console only for a developer terminal; containers have no TTY -> JSON
            json_logs=settings.app_env != "local" or not sys.stdout.isatty(),
            service="ekip-api",
            environment=settings.app_env,
        )
        log = get_logger("ekip.api")
        log.info("startup", environment=settings.app_env)
        yield  # later: open/close DB, Redis and OpenSearch pools here
        if owns_provider:
            provider.shutdown()  # flush spans still waiting in the batch exporter
        log.info("shutdown")

    is_prod = settings.app_env == "production"
    app = FastAPI(
        title="EKIP API",
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
    )
    app.state.settings = settings
    app.state.retriever = retriever or InMemoryRetriever(DEMO_CHUNKS)  # where adapters are chosen
    app.state.readiness = ReadinessService(
        build_default_probes(settings) if probes is None else probes
    )
    app.add_middleware(RequestContextMiddleware)
    # Instrument LAST: the last middleware added is the outermost, so the request span already
    # exists when RequestContextMiddleware writes its access log (that is how logs get trace_id).
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        exclude_spans=["receive", "send"],  # no per-message ASGI spans: one span per request
        excluded_urls="/health$,/ready$",  # probes fire every few seconds: keep traces useful
        # the instrumentation sets the URL (with ?q=...) after the span starts, so scrub here
        server_request_hook=lambda span, _scope: scrub_url_attributes(span),
    )
    app.include_router(health.router)
    app.include_router(ready.router)
    app.include_router(search.router)
    return app
