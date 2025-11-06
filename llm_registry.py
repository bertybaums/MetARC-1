# ---------- xAI Grok ----------
# ---------- Google Gemini ----------
from __future__ import annotations

# llm_registry.py
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from openai import OpenAI
from pydantic import BaseModel, conlist, constr  # PARITY: same as OpenAIAdapter

LLMName = Literal["ollama", "openai", "gemini", "claude", "grok"]

MARCOUTPUT = {
    "title": "ARC AGI Task LLM Output",
    "description": "Schema for the output of an an LLM predicting solutions for ARC AGI tasks.",
    "type": "object",
    "required": ["output_grid", "reasoning"],
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explanation or reasoning behind the prediction.",
            "minLength": 50,
        },
        "output_grid": {
            "type": "array",
            "description": "The predicted 2D grid as an array of arrays of integers.",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}


# ---------- Base interface ----------
class LLMAdapter(ABC):
    """
    Adapter contract. Implementations should return a single string
    containing the model's full text output (or a JSON string when using
    structured outputs). Optionally accept a JSON Schema for structured outputs.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str: ...


# ---------- Helper: uniform system/user prompt packing ----------
def default_chat_messages(prompt: str) -> List[Dict[str, str]]:
    # Most chat APIs accept a list of role/content messages.
    return [
        {"role": "system", "content": "You are a precise ARC puzzle solver."},
        {"role": "user", "content": prompt},
    ]


# ---------- Ollama (local server) ----------
class OllamaAdapter(LLMAdapter):
    """
    Uses Ollama's local HTTP API:
      POST http://localhost:11434/api/generate
    Env:
      OLLAMA_BASE (default http://localhost:11434)
    """

    def __init__(self, base: Optional[str] = None):
        import requests  # local import to avoid hard deps if unused

        self._requests = requests
        self.base = base or os.getenv("OLLAMA_BASE", "http://localhost:11434")

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_format_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base}/api/generate"
        if response_format_schema:
            schema_str = json.dumps(response_format_schema)
            prompt = (
                f"{prompt}\n\n"
                "Respond ONLY with a JSON object that strictly conforms to this schema.\n"
                "Do not include code fences or extra commentary.\n"
                f"JSON_SCHEMA: {schema_str}"
            )

        options: Dict[str, Any] = {}
        if (temperature := kwargs.get("temperature")) is not None:
            options["temperature"] = float(temperature)
        if (top_p := kwargs.get("top_p")) is not None:
            options["top_p"] = float(top_p)
        if (repeat_penalty := kwargs.get("repeat_penalty")) is not None:
            options["repeat_penalty"] = float(repeat_penalty)
        if (num_ctx := kwargs.get("num_ctx")) is not None:
            options["num_ctx"] = int(num_ctx)
        if (max_tokens := kwargs.get("max_tokens")) is not None:
            options["num_predict"] = int(max_tokens)

        payload = {
            "model": model or "llama3.1",
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        resp = self._requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("response", "") or ""

        reasoning_text = ""
        json_candidate = raw_text.strip()
        if json_candidate.startswith("```"):
            json_candidate = json_candidate.strip()
            if json_candidate.startswith("```json"):
                json_candidate = json_candidate[len("```json") :].strip()
            elif json_candidate.startswith("```"):
                json_candidate = json_candidate[len("```") :].strip()
            if json_candidate.endswith("```"):
                json_candidate = json_candidate[:-3].strip()

        try:
            parsed_payload = json.loads(json_candidate)
        except json.JSONDecodeError:
            parsed_payload = None

        if isinstance(parsed_payload, dict):
            reasoning_field = parsed_payload.get("reasoning")
            if isinstance(reasoning_field, str):
                reasoning_text = reasoning_field.strip()
            raw_text = json.dumps(parsed_payload)

        return raw_text, reasoning_text


# ---------- OpenAI (Responses API + Structured Outputs) ----------
# llm_registry.py (OpenAIAdapter)


class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self._client = OpenAI(
            api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE")
        )

    def _as_structured_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": (schema.get("title") or "response_schema").replace(" ", "_"),
                "strict": True,
                "schema": schema,
            },
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        model = model or "gpt-5"

        Cell = constr()
        Row = conlist(Cell, min_length=1)
        Grid = conlist(Row, min_length=1)

        class ARCOutput(BaseModel):
            reasoning: constr(min_length=50)
            output_grid: Grid

        # ✅ Pydantic-first: Responses.parse + text_format
        response = self._client.responses.parse(
            model=model,
            input=prompt,  # string or messages[]
            text_format=ARCOutput,  # <- Pydantic class
            reasoning={"effort": "high", "summary": "detailed"},
        )
        parsed = response.output_parsed  # Pydantic instance
        reasoning_text = ""
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                for summary_item in item.summary:
                    if summary_item.type == "summary_text":
                        reasoning_text += summary_item.text + "\n"

        # Return a JSON string to keep your existing downstream logic
        return parsed.model_dump_json(), reasoning_text


# ---------- Google Gemini ----------


class GeminiAdapter(LLMAdapter):
    """
    Google Gen AI (google-genai) adapter with structured output parity.

    Env: GOOGLE_API_KEY or GEMINI_API_KEY
    Returns: (raw_json_string, reasoning_text), same as OpenAIAdapter.
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)

    # --- helpers -------------------------------------------------------------

    _TYPE_MAP = {
        "object": "OBJECT",
        "array": "ARRAY",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
    }

    def _to_genai_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        # Convert “JSON-Schema-ish” dict to google-genai’s schema dialect.
        if not isinstance(schema, dict):
            return schema
        out: Dict[str, Any] = {}

        t = schema.get("type")
        if isinstance(t, str):
            out["type"] = self._TYPE_MAP.get(t, t.upper())

        props = schema.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: self._to_genai_schema(v) for k, v in props.items()}

        if "items" in schema:
            out["items"] = self._to_genai_schema(schema["items"])

        if "required" in schema and isinstance(schema["required"], list):
            out["required"] = list(schema["required"])

        # Strip unsupported keys that trigger 400s on Gemini.
        for bad in ("additionalProperties", "additional_properties"):
            out.pop(bad, None)

        # Pass through a few harmless hints (ignored if unknown)
        for k in (
            "description",
            "format",
            "minimum",
            "maximum",
            "minItems",
            "maxItems",
            "enum",
        ):
            if k in schema:
                out[k] = schema[k]
        return out

    def _build_config(
        self,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_instruction: Optional[str] = None,
        response_format_schema: Optional[Union[Dict[str, Any], Any]] = None,
    ):
        types = self._types
        cfg: Dict[str, Any] = {}

        if temperature is not None:
            cfg["temperature"] = float(temperature)
        if max_tokens is not None:
            cfg["max_output_tokens"] = int(max_tokens)
        if top_p is not None:
            cfg["top_p"] = float(top_p)
        if top_k is not None:
            cfg["top_k"] = int(top_k)
        if system_instruction:
            cfg["system_instruction"] = str(system_instruction)

        if response_format_schema:
            schema_arg = response_format_schema
            if isinstance(schema_arg, dict):
                schema_arg = self._to_genai_schema(schema_arg)
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema_arg  # Pydantic class OR converted dict
        return types.GenerateContentConfig(**cfg) if cfg else None

    # --- main call -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_format_schema: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> str:
        model_name = model or "gemini-2.5-pro"  # PARITY: use a strong 2.5 default

        # PARITY: If caller didn’t provide a schema, mirror OpenAIAdapter by defining
        # the same ARCOutput here and requesting structured output automatically.
        if response_format_schema is None:
            Cell = constr()
            Row = conlist(Cell, min_length=1)
            Grid = conlist(Row, min_length=1)

            class ARCOutput(BaseModel):
                reasoning: constr(min_length=50)
                output_grid: Grid

            response_format_schema = ARCOutput  # Pydantic class (best path on Gemini) :contentReference[oaicite:1]{index=1}

        config = self._build_config(
            response_format_schema=response_format_schema,
        )

        # Call the new SDK (models.generate_content). :contentReference[oaicite:2]{index=2}
        response = self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        print(response)

        # Prefer parsed when a schema was supplied; otherwise .text. :contentReference[oaicite:3]{index=3}
        reasoning_text = ""
        raw_text = ""

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            print("Is pydantic")
            if hasattr(parsed, "model_dump_json"):  # Pydantic instance
                raw_text = parsed.model_dump_json()
                r = getattr(parsed, "reasoning", None)
                if isinstance(r, str):
                    reasoning_text = r.strip()
            else:
                # dict/list
                try:
                    raw_text = json.dumps(parsed)
                    if isinstance(parsed, dict):
                        r = parsed.get("reasoning")
                        if isinstance(r, str):
                            reasoning_text = r.strip()
                except Exception:
                    raw_text = getattr(response, "text", "") or ""
        else:
            print("Is not pydantic")
            raw_text = getattr(response, "text", "") or ""
            # try to extract reasoning if JSON
            if raw_text:
                try:
                    payload = json.loads(raw_text)
                    if isinstance(payload, dict) and isinstance(
                        payload.get("reasoning"), str
                    ):
                        reasoning_text = payload["reasoning"].strip()
                        raw_text = json.dumps(payload)
                except json.JSONDecodeError:
                    pass
            if not raw_text:
                # last resort: stitch parts
                parts: List[str] = []
                for cand in getattr(response, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        t = getattr(part, "text", None)
                        if t:
                            parts.append(t)
                raw_text = "".join(parts)

        return raw_text, reasoning_text


# ---------- Anthropic Claude ----------
class ClaudeAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

        self._tools = [
            {
                "name": "emit_json",
                "description": "Return the final answer strictly as structured JSON.",
                "input_schema": MARCOUTPUT,
            }
        ]
        self._tool_choice = {"type": "tool", "name": "emit_json"}

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        import json

        model = model or "claude-3-5-sonnet-20240620"

        system_instruction = kwargs.get(
            "system_instruction",
            "You are a precise ARC puzzle solver. "
            "Think quietly. When and only when you have the FINAL answer, call the tool emit_json "
            "EXACTLY ONCE with BOTH fields: reasoning (<= 600 chars) and output_grid (2D int array). "
            "Do not stream partial tool inputs. Do not emit any text outside the tool.",
        )

        req = dict(
            model=model,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
            tools=self._tools,
            tool_choice=self._tool_choice,
            max_tokens=int(kwargs.get("max_tokens", 2048)),
            temperature=float(kwargs.get("temperature", 0)),
        )
        if "top_p" in kwargs:
            req["top_p"] = float(kwargs["top_p"])
        if "top_k" in kwargs:
            req["top_k"] = int(kwargs["top_k"])

        # --- STREAM to bypass 10-minute hard limit ---
        with self._client.messages.stream(**req) as stream:
            final = stream.get_final_message()

        # Safety check
        if getattr(final, "stop_reason", None) == "safety":
            raise RuntimeError("Claude blocked the prompt for safety reasons.")

        # Extract tool payload from the streamed final message
        tool_payload = None
        tool_use_id = None
        for block in final.content or []:
            btype = getattr(block, "type", None) or (
                isinstance(block, dict) and block.get("type")
            )
            if btype == "tool_use":
                name = getattr(block, "name", None) or (
                    isinstance(block, dict) and block.get("name")
                )
                if name == "emit_json":
                    tool_use_id = getattr(block, "id", None) or (
                        isinstance(block, dict) and block.get("id")
                    )
                    tool_payload = getattr(block, "input", None) or (
                        isinstance(block, dict) and block.get("input")
                    )
                    break

        if tool_payload is None:
            # Best-effort fallback: parse any text content as JSON
            text_parts = []
            for block in final.content or []:
                t = getattr(block, "text", None) or (
                    isinstance(block, dict) and block.get("text")
                )
                if isinstance(t, str):
                    text_parts.append(t)
            fallback = "".join(text_parts).strip()
            try:
                parsed = json.loads(fallback.strip("` \n"))
                if (
                    isinstance(parsed, dict)
                    and "output_grid" in parsed
                    and "reasoning" in parsed
                ):
                    tool_payload = parsed
            except Exception:
                pass

        if tool_payload is None:
            raise RuntimeError(
                "Claude did not return the required tool payload (emit_json)."
            )

        # (Optional but recommended) validate before returning
        def _validate_arc_payload(payload):
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Non-object JSON: {type(payload).__name__}: {payload!r}"
                )
            if "output_grid" not in payload or "reasoning" not in payload:
                raise ValueError(
                    f"Missing required keys; got {sorted(payload.keys())!r}"
                )
            og = payload["output_grid"]
            if not (
                isinstance(og, list)
                and og
                and all(isinstance(r, list) and r for r in og)
            ):
                raise ValueError("output_grid must be a non-empty 2D array")
            for r in og:
                for c in r:
                    if not isinstance(c, str):
                        raise ValueError(
                            f"Grid cell must be int, got {type(c).__name__}: {c!r}"
                        )
            rsn = payload["reasoning"]
            if not (isinstance(rsn, str) and 50 <= len(rsn)):
                raise ValueError("reasoning must be 50–600 chars")
            return payload

        try:
            payload = _validate_arc_payload(tool_payload)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON shape from Claude: {e}")

        raw_json_string = json.dumps(payload, ensure_ascii=False)
        reasoning_text = (
            payload.get("reasoning", "")
            if isinstance(payload.get("reasoning"), str)
            else ""
        )
        return raw_json_string, reasoning_text


# ---------- Registry ----------
def get_llm_adapter(name: LLMName, **kwargs) -> LLMAdapter:
    name = name.lower()
    if name == "ollama":
        return OllamaAdapter(**kwargs)
    if name == "openai":
        return OpenAIAdapter(**kwargs)
    if name == "gemini":
        return GeminiAdapter(**kwargs)
    if name == "claude":
        return ClaudeAdapter(**kwargs)
    if name == "grok":
        return GrokAdapter(**kwargs)
    raise ValueError(f"Unknown LLM adapter: {name}")
