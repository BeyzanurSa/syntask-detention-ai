"""What the LLM is allowed to return.

The model only *extracts*. Dates stay as raw text ("perşembe", "haftaya bir ara");
code resolves them. No field here is computed (no last free day, no fee).

Every critical field carries `evidence`: a verbatim quote from the message. Code
checks that the quote really is in the text and that it contains the right kind
of keyword (gate-out words for gate_out, return words for planned_return, ...).
A field whose evidence fails is dropped, i.e. treated as "not mentioned".

Key order matters: llama.cpp's grammar emits keys in schema order, so `evidence`
comes before `raw`. The model first quotes the sentence, then copies the date.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateMention(_Strict):
    evidence: str = Field(description="Verbatim quote from the message that states this date.")
    raw: str = Field(description="The date expression exactly as written. Never converted.")


class FreeTime(_Strict):
    evidence: str
    days: int = Field(ge=0, le=60)
    basis: Literal["calendar", "business", "unspecified"]


class RateTier(_Strict):
    usd_per_day: float = Field(gt=0)
    first_n_days: int | None = Field(
        description='Only for "first N days" tiers. null for a flat rate or "thereafter".'
    )


class Rates(_Strict):
    evidence: str
    tiers: list[RateTier] = Field(min_length=1)


class ContainerMention(_Strict):
    container_no: str = Field(description="Exactly as written, spaces/dashes included.")
    gate_out: DateMention | None = Field(
        description="Only from explicit gate-out / gated out / released / picked up wording."
    )
    at_terminal_evidence: str | None = Field(
        description="Quote ONLY if the message explicitly says the container is still at the terminal."
    )
    free_time: FreeTime | None
    rates: Rates | None
    planned_return: DateMention | None = Field(
        description="Only from return / iade wording. Vague expressions are copied as written."
    )


class MessageExtraction(_Strict):
    # No "category" field on purpose: grammar fixes key order, and a leading
    # category made the model commit to "unrelated" before reading. Relevance is
    # decided in code (container-number regex); an empty list = nothing relevant.
    containers: list[ContainerMention]


def llm_json_schema() -> dict[str, Any]:
    """Pydantic schema with $refs inlined, for llama.cpp's JSON-schema grammar."""
    schema = MessageExtraction.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k not in ("title",)}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)
