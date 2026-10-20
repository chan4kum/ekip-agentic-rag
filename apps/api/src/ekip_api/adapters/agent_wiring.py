"""Build the agent chat streamer from settings (used only by the composition root)."""

from collections.abc import Awaitable, Callable

from ekip_agents import (
    Budget,
    LLMGroundednessChecker,
    LLMPlanner,
    LLMQueryRewriter,
    RuleBasedRouter,
    build_agent_graph,
    create_checkpointer,
)
from ekip_config.settings import Settings
from ekip_contracts.chat import ChatStreamer
from ekip_contracts.retrieval import Retriever
from ekip_llm_gateway import OpenAIChatModel

from ekip_api.adapters.agent_chat import AgentChatStreamer
from ekip_api.adapters.conversation_wiring import build_conversation_store

Hook = Callable[[], Awaitable[None]]


def build_agent_streamer(
    settings: Settings, retriever: Retriever
) -> tuple[ChatStreamer, list[Hook], list[Hook]]:
    """Returns the streamer plus startup hooks (open) and shutdown hooks (close)."""
    if settings.llm_api_key is None or not settings.llm_model:
        # fail at startup, not on the first user question
        raise ValueError("CHAT_BACKEND=agent requires LLM_MODEL and LLM_API_KEY")
    model = OpenAIChatModel(
        settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        model=settings.llm_model,
        limit_field=settings.llm_limit_field,
        send_temperature=settings.llm_send_temperature,
    )
    if settings.agent_approval:
        # a paused run needs someone to approve it, and the chat API has no endpoint for that yet
        raise ValueError("AGENT_APPROVAL is not available through the chat API in this phase")
    store, deferred = build_conversation_store(settings)
    saver, checkpointer = create_checkpointer(
        settings.agent_checkpointer, str(settings.database_url)
    )
    budget = Budget(
        max_steps=settings.agent_max_steps,
        max_retrievals=settings.agent_max_retrievals,
        max_model_calls=settings.agent_max_model_calls,
        max_seconds=settings.agent_max_seconds,
    )
    graph = build_agent_graph(
        router=RuleBasedRouter(),
        retriever=retriever,
        model=model,
        store=store,
        rewriter=LLMQueryRewriter(model),
        planner=(
            LLMPlanner(model, max_subtasks=settings.agent_max_subtasks)
            if settings.agent_multistep
            else None
        ),
        grounding=LLMGroundednessChecker(model) if settings.agent_groundedness else None,
        budget=budget,
        max_steps=budget.max_steps,
        checkpointer=saver,
    )
    openers: list[Hook] = [deferred.open] if deferred is not None else []
    closers: list[Hook] = [model.aclose]
    if deferred is not None:
        closers.append(deferred.aclose)
    if checkpointer is not None:
        openers.append(checkpointer.open)
        closers.append(checkpointer.aclose)
    return AgentChatStreamer(graph, checkpointed=saver is not None), openers, closers
