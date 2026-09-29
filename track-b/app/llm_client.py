import json
import logging
import asyncio
import random
from typing import Any, cast

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.tools import TOOL_DEFINITIONS, execute_tool
from app.cache import response_cache
from app.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, factual AI assistant.

Rules you must follow:
1. If the user's question relates to retrieved context provided to you, base your answer
   primarily on that context. Do not use outside knowledge to contradict it.
2. If no relevant context is provided and you don't know the answer, say so honestly
   instead of guessing.
3. When you use a tool, wait for its result before answering — never fabricate a tool result.
4. Keep answers concise and directly address the question asked.
5. If asked to produce structured data, respond with valid JSON only, matching the
   requested schema exactly — no extra commentary, no markdown code fences.
"""


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0
        )

        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
        elif self.provider in ("vllm", "ollama"):
            self.client = AsyncOpenAI(base_url=settings.VLLM_BASE_URL, api_key="not-needed")
            self.model = settings.VLLM_MODEL
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

        # Fallback client
        self._fallback_client = None
        self._fallback_model = None
        if settings.FALLBACK_PROVIDER:
            self._setup_fallback()

    def _setup_fallback(self):
        provider = settings.FALLBACK_PROVIDER
        if provider in ("vllm", "ollama"):
            self._fallback_client = AsyncOpenAI(
                base_url=settings.FALLBACK_VLLM_BASE_URL, api_key="not-needed"
            )
            self._fallback_model = settings.FALLBACK_VLLM_MODEL
            logger.info(f"Fallback provider configured: {provider} ({self._fallback_model})")

    def _build_messages(
        self, user_message: str, history: list[dict] | None = None,
        system_prompt: str | None = None
    ) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _call_with_retry(self, client, model, **kwargs) -> Any:
        last_error = None
        for attempt in range(settings.MAX_RETRIES):
            try:
                result = await client.chat.completions.create(model=model, **kwargs)
                self._circuit_breaker.record_success()
                return result
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError) as e:
                last_error = e
                jitter = random.uniform(0.1, 0.4)
                wait_time = min(
                    settings.RETRY_BACKOFF_BASE * (2 ** attempt),
                    settings.RETRY_MAX_WAIT,
                ) + jitter
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{settings.MAX_RETRIES}): {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                self._circuit_breaker.record_failure()
                await asyncio.sleep(wait_time)
            except openai.APIStatusError as e:
                if e.status_code in (429, 500, 502, 503, 504):
                    last_error = e
                    jitter = random.uniform(0.1, 0.4)
                    wait_time = min(
                        settings.RETRY_BACKOFF_BASE * (2 ** attempt),
                        settings.RETRY_MAX_WAIT,
                    ) + jitter
                    logger.warning(
                        f"LLM API error {e.status_code} (attempt {attempt + 1}/{settings.MAX_RETRIES}): {e}. "
                        f"Retrying in {wait_time:.1f}s"
                    )
                    self._circuit_breaker.record_failure()
                    await asyncio.sleep(wait_time)
                else:
                    self._circuit_breaker.record_failure()
                    raise
            except Exception:
                self._circuit_breaker.record_failure()
                raise
        raise last_error

    async def _call_with_fallback(self, **kwargs) -> Any:
        if not self._circuit_breaker.allow_request():
            if self._fallback_client:
                logger.warning("Circuit open, routing request to fallback provider")
                return await self._fallback_client.chat.completions.create(
                    model=self._fallback_model, **kwargs
                )
            raise Exception("Primary LLM circuit breaker is open and no fallback configured")

        try:
            return await self._call_with_retry(self.client, self.model, **kwargs)
        except Exception as e:
            logger.error(f"Primary provider failed: {e}")
            if self._fallback_client:
                logger.info("Falling back to secondary provider")
                return await self._fallback_client.chat.completions.create(
                    model=self._fallback_model, **kwargs
                )
            raise

    async def acreate(self, **kwargs) -> Any:
        """Raw chat completion returning the full response object (usage included)."""
        return await self._call_with_fallback(**kwargs)

    async def chat(
        self,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        temp = 0.2 if temperature is None else temperature
        top_p_val = 1.0 if top_p is None else top_p
        tokens = 800 if max_tokens is None else max_tokens

        messages = self._build_messages(user_message, history, system_prompt)

        cached = response_cache.get(messages, self.model, temperature=temp, top_p=top_p_val)
        if cached is not None:
            return cached

        response = await self._call_with_fallback(
            messages=messages,
            temperature=temp,
            top_p=top_p_val,
            max_tokens=tokens,
        )
        result = response.choices[0].message.content
        response_cache.set(messages, self.model, result, temperature=temp, top_p=top_p_val)
        return result

    async def chat_structured(
        self,
        user_message: str,
        context: str = "",
        schema_instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        temp = 0.0 if temperature is None else temperature
        tokens = 800 if max_tokens is None else max_tokens

        default_schema = """
        Respond with ONLY a JSON object in exactly this shape, no other text:
        {
        "answer": "<your answer as a string>",
        "sources": [{"document": "<filename>", "chunk_id": "<id>"}],
        "confidence": <float between 0 and 1>
        }
        If you used no retrieved context, return an empty "sources" array.
        """
        instructions = schema_instructions or default_schema
        full_system = SYSTEM_PROMPT + instructions

        user_content = user_message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{user_message}"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        cached = response_cache.get(messages, self.model, temperature=temp)
        if cached is not None:
            return cached

        response = await self._call_with_fallback(
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        parsed = await self._parse_json_with_repair(raw, messages)
        response_cache.set(messages, self.model, parsed, temperature=temp)
        return parsed

    async def chat_with_tools(
        self,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_iterations: int = 5,
    ) -> dict:
        temp = 0.2 if temperature is None else temperature
        tokens = 800 if max_tokens is None else max_tokens

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for _ in range(max_iterations):
            response = await self._call_with_fallback(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                return {"answer": message.content, "tool_calls_made": tool_calls_made}

            function_tool_calls = [cast(Any, tc) for tc in message.tool_calls]
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in function_tool_calls
                ],
            })

            for tc in function_tool_calls:
                tool_calls_made.append(tc.function.name)
                result = execute_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        final = await self._call_with_fallback(
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return {"answer": final.choices[0].message.content, "tool_calls_made": tool_calls_made}

    async def _parse_json_with_repair(self, raw: str, original_messages: list[dict]) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            repair_messages = original_messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Return ONLY the corrected valid JSON object, nothing else.",
                },
            ]
            response = await self._call_with_fallback(
                messages=repair_messages,
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw_retry = response.choices[0].message.content
            return json.loads(raw_retry)


llm_client = LLMClient()

