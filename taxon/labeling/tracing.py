"""Optional Langfuse tracing of labeling calls, on only when its keys are set."""

from __future__ import annotations

from taxon.labeling.env import Secrets
from taxon.labeling.llm import LLMRequest, LLMResult


class LangfuseTracer:
    def __init__(self, client):
        self.client = client

    def record(self, request: LLMRequest, result: LLMResult) -> None:
        usage = result.usage
        observation = self.client.start_observation(
            name=f"labeling.{request.purpose}",
            as_type="generation",
            model=request.config.model,
            input=request.user,
            output=result.text,
            metadata={
                "vendor": request.config.vendor,
                "status": result.status.value,
                "key": request.key,
                "has_image": request.image_path is not None,
            },
            usage_details={
                "input": usage.input_tokens,
                "cached": usage.cached_tokens,
                "cache_write": usage.cache_write_tokens,
                "output": usage.output_tokens,
            },
            cost_details={"total": result.cost},
        )
        observation.end()

    def flush(self) -> None:
        self.client.flush()


def build_tracer(secrets: Secrets, host: str | None = None) -> LangfuseTracer | None:
    public_key = secrets.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secrets.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    from langfuse import Langfuse

    return LangfuseTracer(Langfuse(public_key=public_key, secret_key=secret_key, host=host))
