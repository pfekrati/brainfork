"""
Brainfork ↔ Microsoft Agent Framework integration
==================================================

This example shows how to use Brainfork as a routing middleware in front of
the Microsoft Agent Framework (https://learn.microsoft.com/en-us/agent-framework/).

The Agent Framework's `ChatAgent` is constructed with a `chat_client` (e.g.
`AzureOpenAIChatClient`). That client is bound to a single deployment, so the
agent always talks to the same model. Brainfork's job is to inspect the
incoming conversation and pick the *right* deployment per turn — which means
we need to plug Brainfork in at the chat-client layer, not above the agent.

Two integration patterns are shown:

  1. `route_then_build` — pick a model with Brainfork up front, then build a
     ChatAgent bound to that deployment. Simple, but the routing decision is
     frozen for the lifetime of the agent.

  2. `BrainforkChatClient` — a thin wrapper that implements the parts of the
     Agent Framework chat-client surface the agent calls, and re-routes on
     every turn. This is the actual "middleware" — one agent, many models.

Install:
    pip install brainfork agent-framework

Run:
    python examples/agent_framework_integration.py
"""

import asyncio
from typing import Any

from brainfork import ModelRouter, ModelConfig, AuthConfig, UseCase

# Agent Framework imports. The public surface is still settling; this example
# targets the `agent-framework` package shown in the Microsoft Learn quickstart.
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient


# ---------------------------------------------------------------------------
# Shared Brainfork setup
# ---------------------------------------------------------------------------

def build_router() -> ModelRouter:
    """Configure the deployments Brainfork is allowed to route between."""
    auth = AuthConfig(api_key="your-api-key")
    endpoint = "https://your-endpoint.openai.azure.com/"

    models = {
        "gpt-4o-mini": ModelConfig(
            endpoint=endpoint,
            deployment_name="gpt-4o-mini",
            api_version="2024-10-21",
            auth=auth,
        ),
        "gpt-4o": ModelConfig(
            endpoint=endpoint,
            deployment_name="gpt-4o",
            api_version="2024-10-21",
            auth=auth,
        ),
        "o1-mini": ModelConfig(
            endpoint=endpoint,
            deployment_name="o1-mini",
            api_version="2024-10-21",
            auth=auth,
        ),
    }

    use_cases = [
        UseCase(
            name="cheap_chat",
            description="Short, factual answers and small-talk where cost matters more than depth.",
            model_name="gpt-4o-mini",
            keywords=["hi", "hello", "what is", "define"],
            min_confidence=0.6,
        ),
        UseCase(
            name="reasoning",
            description="Multi-step reasoning, math, planning, code review.",
            model_name="o1-mini",
            keywords=["why", "prove", "derive", "plan", "debug"],
            min_confidence=0.75,
        ),
        UseCase(
            name="general_writing",
            description="Drafting, summarization, longer-form responses.",
            model_name="gpt-4o",
            keywords=["write", "summarize", "draft", "explain"],
            min_confidence=0.7,
        ),
    ]

    return ModelRouter(
        models=models,
        use_cases=use_cases,
        default_model="gpt-4o-mini",
        routing_model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Pattern 1: route once, then build a ChatAgent
# ---------------------------------------------------------------------------
#
# Easiest path. You call Brainfork up front to decide which deployment to
# use, then construct an Agent Framework `ChatAgent` against that deployment.
# Good for short-lived agents handling a single request.

async def route_then_build(user_message: str) -> str:
    router = build_router()

    messages = [{"role": "user", "content": user_message}]
    result = await router.route_conversation(messages)
    chosen = router.models[result.model_name]

    print(f"[brainfork] chose {result.model_name} "
          f"(confidence={result.confidence:.2f}, reason={result.reasoning})")

    chat_client = AzureOpenAIChatClient(
        endpoint=chosen.endpoint,
        deployment_name=chosen.deployment_name,
        api_version=chosen.api_version,
        api_key=chosen.auth.api_key,
    )

    agent = ChatAgent(
        chat_client=chat_client,
        name="brainfork-routed-agent",
        instructions="You are a helpful assistant.",
    )

    response = await agent.run(user_message)
    return response.text


# ---------------------------------------------------------------------------
# Pattern 2: BrainforkChatClient — true per-turn routing middleware
# ---------------------------------------------------------------------------
#
# A `ChatAgent` calls its `chat_client` once per turn. By wrapping that call
# we can re-route on every turn based on the *current* conversation state.
#
# The Agent Framework chat-client contract evolves; the relevant entry points
# are `get_response(...)` (one-shot) and `get_streaming_response(...)`
# (streaming). We implement both by delegating to a freshly-built underlying
# Azure client chosen by Brainfork. We do NOT subclass the framework's base
# class here — a duck-typed wrapper keeps the example version-independent.

class BrainforkChatClient:
    """Agent Framework-compatible chat client that routes each call via Brainfork."""

    def __init__(self, router: ModelRouter):
        self._router = router
        self._client_cache: dict[str, AzureOpenAIChatClient] = {}

    def _client_for(self, model_name: str) -> AzureOpenAIChatClient:
        if model_name not in self._client_cache:
            cfg = self._router.models[model_name]
            self._client_cache[model_name] = AzureOpenAIChatClient(
                endpoint=cfg.endpoint,
                deployment_name=cfg.deployment_name,
                api_version=cfg.api_version,
                api_key=cfg.auth.api_key,
            )
        return self._client_cache[model_name]

    @staticmethod
    def _to_router_messages(messages: Any) -> list[dict[str, str]]:
        """Normalize Agent Framework ChatMessage objects → plain dicts for Brainfork."""
        out: list[dict[str, str]] = []
        for m in messages:
            role = getattr(m, "role", None) or m["role"]
            role = getattr(role, "value", role)
            content = getattr(m, "text", None) or getattr(m, "content", None) or m.get("content", "")
            out.append({"role": str(role), "content": str(content)})
        return out

    async def get_response(self, messages, **kwargs):
        routed = await self._router.route_conversation(self._to_router_messages(messages))
        print(f"[brainfork] turn routed → {routed.model_name} ({routed.confidence:.2f})")
        return await self._client_for(routed.model_name).get_response(messages, **kwargs)

    async def get_streaming_response(self, messages, **kwargs):
        routed = await self._router.route_conversation(self._to_router_messages(messages))
        print(f"[brainfork] turn routed → {routed.model_name} ({routed.confidence:.2f})")
        async for chunk in self._client_for(routed.model_name).get_streaming_response(messages, **kwargs):
            yield chunk


async def per_turn_routing_demo() -> None:
    router = build_router()
    agent = ChatAgent(
        chat_client=BrainforkChatClient(router),
        name="brainfork-middleware-agent",
        instructions="You are a helpful assistant.",
    )

    # Each turn is routed independently — cheap models for chit-chat,
    # reasoning models when the user asks something hard.
    for prompt in [
        "Hi! What's your name?",
        "Prove that the square root of 2 is irrational, step by step.",
        "Now summarize that proof in two sentences for a 10-year-old.",
    ]:
        print(f"\nUser: {prompt}")
        response = await agent.run(prompt)
        print(f"Agent: {response.text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Pattern 1: route once, then build the agent")
    print("=" * 60)
    answer = await route_then_build("Write a haiku about distributed systems.")
    print(f"Agent: {answer}")

    print("\n" + "=" * 60)
    print("Pattern 2: BrainforkChatClient — per-turn routing")
    print("=" * 60)
    await per_turn_routing_demo()


if __name__ == "__main__":
    asyncio.run(main())
