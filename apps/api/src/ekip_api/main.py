import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from ekip_authorization.rbac import RbacAuthorizer
from ekip_config.settings import Settings, load_settings
from ekip_contracts.authorization import Authorizer
from ekip_contracts.chat import ChatStreamer
from ekip_contracts.health import DependencyProbe
from ekip_contracts.identity import TokenValidator
from ekip_contracts.ratelimit import RateLimits
from ekip_contracts.retrieval import Retriever
from ekip_identity.jwks import CachedJwks, httpx_fetcher
from ekip_identity.validator import JwtTokenValidator
from ekip_observability import EkipMetrics, install_metrics
from ekip_observability.logging import configure_logging, get_logger
from ekip_observability.tracing import build_tracer_provider, scrub_url_attributes
from ekip_ratelimit.redis_bucket import RedisRateLimiter
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from redis.asyncio import Redis

from ekip_api.adapters.agent_wiring import build_agent_streamer
from ekip_api.adapters.probes import build_default_probes
from ekip_api.adapters.redis_idempotency import RedisIdempotencyStore
from ekip_api.adapters.retrieval_wiring import build_retriever
from ekip_api.adapters.stub_chat import StubChatStreamer
from ekip_api.body_limit import MaxBodySizeMiddleware
from ekip_api.deadline import RequestDeadlineMiddleware
from ekip_api.errors import install_error_handlers
from ekip_api.middleware import RequestContextMiddleware
from ekip_api.routes import chat, health, ready, search
from ekip_api.services.idempotency import IdempotencyService
from ekip_api.services.readiness import ReadinessService


def _redis_rate_limits(settings: Settings, redis: Redis) -> RateLimits:
    return RateLimits(
        user=RedisRateLimiter(
            redis,
            capacity=settings.rate_limit_user_burst,
            refill_per_second=settings.rate_limit_user_per_minute / 60,
            prefix="ekip:rl:user",
        ),
        tenant=RedisRateLimiter(
            redis,
            capacity=settings.rate_limit_tenant_burst,
            refill_per_second=settings.rate_limit_tenant_per_minute / 60,
            prefix="ekip:rl:tenant",
        ),
        fail_open=settings.rate_limit_fail_open,
    )


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    probes: Sequence[DependencyProbe] | None = None,
    token_validator: TokenValidator | None = None,
    authorizer: Authorizer | None = None,
    rate_limits: RateLimits | None = None,
    idempotency: IdempotencyService | None = None,
    chat_streamer: ChatStreamer | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Composition root: the only place that wires concrete dependencies."""
    settings = settings or load_settings()
    owned_redis: Redis | None = None  # only created when some Redis-backed default is needed
    if rate_limits is None or idempotency is None:
        owned_redis = Redis.from_url(str(settings.redis_url))  # lazy: connects on first use
    if rate_limits is None and owned_redis is not None:
        rate_limits = _redis_rate_limits(settings, owned_redis)
    if idempotency is None and owned_redis is not None:
        idempotency = IdempotencyService(RedisIdempotencyStore(owned_redis))
    owns_provider = tracer_provider is None  # only shut down a provider we built ourselves
    provider = tracer_provider or build_tracer_provider(
        service="ekip-api", environment=settings.app_env, exporter=settings.tracing_exporter
    )

    closers: list[Callable[[], Awaitable[None]]] = []  # shutdown hooks for clients we opened
    openers: list[Callable[[], Awaitable[None]]] = []  # startup hooks (need the event loop)

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
        for open_ in openers:  # resources that need the running event loop (e.g. a DB pool)
            await open_()
        yield
        for close in closers:  # clients the retrieval backend opened
            await close()
        if owned_redis is not None:
            await owned_redis.aclose()
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
    install_error_handlers(app)
    app.state.settings = settings
    if retriever is None:
        retriever, retriever_closers = build_retriever(settings)  # where adapters are chosen
        closers.extend(retriever_closers)
    app.state.retriever = retriever
    if chat_streamer is None:
        if settings.chat_backend == "agent":
            chat_streamer, agent_openers, agent_closers = build_agent_streamer(settings, retriever)
            openers.extend(agent_openers)
            closers.extend(agent_closers)
        else:
            chat_streamer = StubChatStreamer(
                retriever, token_delay_s=settings.chat_stub_token_delay_s
            )
    app.state.chat_streamer = chat_streamer
    app.state.token_validator = token_validator or JwtTokenValidator(
        issuer=str(settings.oidc_issuer).rstrip("/"),
        audience=settings.oidc_audience,
        keys=CachedJwks(httpx_fetcher(str(settings.oidc_jwks_url))),
    )
    app.state.authorizer = authorizer or RbacAuthorizer()
    app.state.rate_limits = rate_limits
    app.state.idempotency = idempotency
    app.state.readiness = ReadinessService(
        build_default_probes(settings) if probes is None else probes
    )
    # Inner to RequestContextMiddleware (added next, so it wraps this): a 504 is still logged and
    # carries the request id. Streaming routes (Gate 2.7) will be exempted and manage their own.
    app.add_middleware(
        RequestDeadlineMiddleware,
        timeout_s=settings.request_timeout_s,
        exempt_prefixes=(
            "/api/v1/chat",
        ),  # streams have their own deadline (settings.chat_timeout_s)
    )
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_request_body_bytes)
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
    app.include_router(chat.router)
    # Outermost: times the whole request; /metrics is 404 without a configured bearer value.
    install_metrics(app, EkipMetrics(), token=settings.metrics_token)
    return app
