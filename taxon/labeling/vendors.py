"""Batch clients for three vendors, each behind the same small interface.

Base URLs are passed explicitly so that a variable in the surrounding shell can never send a key
somewhere else.
"""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import anthropic
import openai
from google import genai
from google.genai import types as genai_types

from taxon.labeling.env import Secrets
from taxon.labeling.llm import LLMRequest, LLMResult, Status, Unreachable, Usage

ANTHROPIC_URL = "https://api.anthropic.com"
OPENAI_URL = "https://api.openai.com/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com"

IMAGE_MEDIA_TYPE = "image/jpeg"
OPENAI_ENDPOINT = "/v1/chat/completions"
_OPENAI_FINISHED = {"completed", "failed", "expired", "cancelled"}
_GOOGLE_FINISHED = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
# A batch can wait up to 48 hours, so a cache must outlive that. Storage is $1 per million tokens
# per hour on the chosen model.
_GOOGLE_CACHE_SECONDS = 172_800
_GOOGLE_CACHE_DOLLARS_PER_TOKEN = 48 * 1.00 / 1_000_000

Charge = Callable[[str, float, str], None]


def _image_base64(request: LLMRequest) -> str | None:
    if request.image_path is None:
        return None
    return base64.b64encode(Path(request.image_path).read_bytes()).decode("ascii")


class AnthropicBatchClient:
    def __init__(self, api_key: str, base_url: str = ANTHROPIC_URL):
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    @staticmethod
    def params(request: LLMRequest) -> dict:
        content: list[dict] = []
        image = _image_base64(request)
        if image is not None:
            source = {"type": "base64", "media_type": IMAGE_MEDIA_TYPE, "data": image}
            content.append({"type": "image", "source": source})
        content.append({"type": "text", "text": request.user})
        return {
            "model": request.config.model,
            "max_tokens": request.config.max_tokens,
            # A batch runs for longer than the default five minutes, so entries get an hour.
            "system": [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": request.config.effort,
                "format": {"type": "json_schema", "schema": request.schema},
            },
        }

    def submit(self, requests: list[LLMRequest]) -> str:
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": r.key, "params": self.params(r)} for r in requests]
        )
        return batch.id

    def poll(self, batch_id: str) -> bool:
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
        except anthropic.APIConnectionError as error:
            raise Unreachable("anthropic") from error
        return batch.processing_status == "ended"

    def results(self, batch_id: str) -> list[LLMResult]:
        return [self._result(row) for row in self.client.messages.batches.results(batch_id)]

    @staticmethod
    def _result(row) -> LLMResult:
        if row.result.type != "succeeded":
            detail = getattr(getattr(row.result, "error", None), "type", row.result.type)
            return LLMResult(row.custom_id, Status.FAILED, error=str(detail))
        message = row.result.message
        usage = Usage(
            input_tokens=message.usage.input_tokens,
            cached_tokens=message.usage.cache_read_input_tokens or 0,
            cache_write_tokens=message.usage.cache_creation_input_tokens or 0,
            output_tokens=message.usage.output_tokens,
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        status = {"refusal": Status.REFUSED, "max_tokens": Status.TRUNCATED}.get(
            message.stop_reason, Status.OK
        )
        return LLMResult(row.custom_id, status, text, usage)


class OpenAIBatchClient:
    def __init__(self, api_key: str, base_url: str = OPENAI_URL):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def body(request: LLMRequest) -> dict:
        content: list[dict] = [{"type": "text", "text": request.user}]
        image = _image_base64(request)
        if image is not None:
            url = f"data:{IMAGE_MEDIA_TYPE};base64,{image}"
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
        return {
            "model": request.config.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": request.config.max_tokens,
            "reasoning_effort": request.config.effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": request.purpose, "strict": True, "schema": request.schema},
            },
            "prompt_cache_key": "taxon-labeling",
        }

    def submit(self, requests: list[LLMRequest]) -> str:
        lines = [
            json.dumps(
                {
                    "custom_id": r.key,
                    "method": "POST",
                    "url": OPENAI_ENDPOINT,
                    "body": self.body(r),
                }
            )
            for r in requests
        ]
        payload = ("\n".join(lines) + "\n").encode()
        uploaded = self.client.files.create(file=("batch.jsonl", payload), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id, endpoint=OPENAI_ENDPOINT, completion_window="24h"
        )
        return batch.id

    def poll(self, batch_id: str) -> bool:
        try:
            batch = self.client.batches.retrieve(batch_id)
        except openai.APIConnectionError as error:
            raise Unreachable("openai") from error
        return batch.status in _OPENAI_FINISHED

    def results(self, batch_id: str) -> list[LLMResult]:
        batch = self.client.batches.retrieve(batch_id)
        results = []
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                content = self.client.files.content(file_id).text
                results.extend(self._result(json.loads(x)) for x in content.splitlines() if x)
        return results

    @staticmethod
    def _result(row: dict) -> LLMResult:
        key = row["custom_id"]
        response = row.get("response")
        if not response or response.get("status_code") != 200:
            error = row.get("error") or (response or {}).get("body", {}).get("error") or {}
            return LLMResult(key, Status.FAILED, error=str(error.get("code", "request failed")))
        body = response["body"]
        details = body["usage"].get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        written = details.get("cache_write_tokens") or 0
        usage = Usage(
            input_tokens=body["usage"]["prompt_tokens"] - cached - written,
            cached_tokens=cached,
            cache_write_tokens=written,
            output_tokens=body["usage"]["completion_tokens"],
        )
        choice = body["choices"][0]
        message = choice["message"]
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            return LLMResult(key, Status.REFUSED, "", usage)
        status = Status.TRUNCATED if choice.get("finish_reason") == "length" else Status.OK
        return LLMResult(key, status, message.get("content") or "", usage)


class GoogleBatchClient:
    def __init__(self, api_key: str, base_url: str = GOOGLE_URL, charge: Charge | None = None):
        self.client = genai.Client(
            api_key=api_key, http_options=genai_types.HttpOptions(base_url=base_url)
        )
        self.charge = charge
        self._caches: dict[str, str | None] = {}

    def _cache_for(self, model: str, system: str) -> str | None:
        """An explicit cache of the shared prefix, or None if the model turns it down.

        Requests still work without one, and the dry run reports how many tokens were cached.
        """
        identity = hashlib.sha256(f"{model}|{system}".encode()).hexdigest()
        if identity not in self._caches:
            try:
                created = self.client.caches.create(
                    model=model,
                    config=genai_types.CreateCachedContentConfig(
                        system_instruction=system, ttl=f"{_GOOGLE_CACHE_SECONDS}s"
                    ),
                )
                self._caches[identity] = created.name
                if self.charge is not None:
                    storage = len(system) / 4 * _GOOGLE_CACHE_DOLLARS_PER_TOKEN
                    self.charge("google", storage, "explicit cache storage")
            except genai.errors.APIError:
                self._caches[identity] = None
        return self._caches[identity]

    @staticmethod
    def line(request: LLMRequest, cache_name: str | None) -> dict:
        parts: list[dict] = []
        image = _image_base64(request)
        if image is not None:
            parts.append({"inlineData": {"mimeType": IMAGE_MEDIA_TYPE, "data": image}})
        parts.append({"text": request.user})
        body: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": request.config.max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": request.schema,
                "thinkingConfig": {"thinkingLevel": request.config.effort.upper()},
            },
        }
        if cache_name:
            body["cachedContent"] = cache_name
        else:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        return {"key": request.key, "request": body}

    def submit(self, requests: list[LLMRequest]) -> str:
        model = requests[0].config.model
        cache_name = self._cache_for(model, requests[0].system)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for request in requests:
                    f.write(json.dumps(self.line(request, cache_name)) + "\n")
            uploaded = self.client.files.upload(
                file=path, config=genai_types.UploadFileConfig(mime_type="jsonl")
            )
        return self.client.batches.create(model=model, src=uploaded.name).name

    def poll(self, batch_id: str) -> bool:
        return self.client.batches.get(name=batch_id).state.name in _GOOGLE_FINISHED

    def results(self, batch_id: str) -> list[LLMResult]:
        job = self.client.batches.get(name=batch_id)
        if job.state.name != "JOB_STATE_SUCCEEDED" or not job.dest or not job.dest.file_name:
            return []
        content = self.client.files.download(file=job.dest.file_name).decode("utf-8")
        return [self._result(json.loads(x)) for x in content.splitlines() if x.strip()]

    @staticmethod
    def _result(row: dict) -> LLMResult:
        key = row["key"]
        response = row.get("response")
        if response is None:
            error = row.get("status") or row.get("error") or {}
            return LLMResult(key, Status.FAILED, error=str(error.get("message", "request failed")))
        meta = response.get("usageMetadata") or {}
        cached = meta.get("cachedContentTokenCount", 0)
        usage = Usage(
            input_tokens=meta.get("promptTokenCount", 0) - cached,
            cached_tokens=cached,
            output_tokens=meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0),
        )
        candidates = response.get("candidates") or []
        if not candidates:
            return LLMResult(key, Status.REFUSED, "", usage)
        reason = candidates[0].get("finishReason", "STOP")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if reason == "MAX_TOKENS":
            return LLMResult(key, Status.TRUNCATED, text, usage)
        if reason != "STOP":
            return LLMResult(key, Status.REFUSED, "", usage)
        return LLMResult(key, Status.OK, text, usage)


def build_clients(secrets: Secrets, vendors: Iterable[str], charge: Charge | None = None) -> dict:
    """The real clients for the vendors in use. Raises MissingKey, naming the key, if one is unset.

    Only the named vendors' keys are needed, so an unused vendor needs no key.
    """
    makers = {
        "anthropic": lambda: AnthropicBatchClient(secrets.require("ANTHROPIC_API_KEY")),
        "openai": lambda: OpenAIBatchClient(secrets.require("OPENAI_API_KEY")),
        "google": lambda: GoogleBatchClient(secrets.require("GEMINI_API_KEY"), charge=charge),
    }
    return {vendor: makers[vendor]() for vendor in vendors}
