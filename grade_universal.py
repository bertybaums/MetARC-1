# grading/grade_universal.py
import json
import re
from typing import Any, Optional

import requests
from llm_adapters.llm_registry import get_llm_adapter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MARCOUTPUT = {
    "title": "ARC AGI Task LLM Output",
    "description": "Schema for the output of an an LLM predicting solutions for ARC AGI tasks.",
    "type": "object",
    "required": ["output_grid", "reasoning"],
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explanation or reasoning behind the prediction. This should be a detailed, chain-of-thought explanation of how the output grid was derived from the input, describing the rules, patterns, or transformations applied.",
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


THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_STRIP_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


LONG_READ_SECS = 15 * 60  # 15 minutes
CONNECT_SECS = 10


def _make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["POST", "GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def _post_with_long_timeout(url, headers, payload, stream=False):
    session = _make_session()
    # NOTE: timeout can be tuple (connect, read); set read very high for long jobs.
    return session.post(
        url,
        headers=headers,
        json=payload,
        timeout=(CONNECT_SECS, LONG_READ_SECS),
        stream=stream,
    )


def extract_think(text: str) -> str:
    """Return all <think>...</think> contents concatenated."""
    blocks = THINK_TAG_RE.findall(text or "")
    return "\n\n".join(b.strip() for b in blocks if b is not None).strip()


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks."""
    return THINK_STRIP_RE.sub("", text or "").strip()


def _parse_json_lenient(s: str):
    """Try normal JSON parse; on failure, try common wrappers."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Strip code fences if present
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
        # Try first {...} slice
        l, r = s.find("{"), s.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(s[l : r + 1])
            except Exception:
                pass
        # Your legacy corrupted wrapper fallback
        try:
            return json.loads(s[2:-1])
        except Exception as e:
            raise e


def format_grid(grid: list[list[int]]) -> str:
    """Formats a 2D list into a visually clear string."""
    return "\n".join(" ".join(map(str, row)) for row in grid)


# --- 0. Helpers -----------------------------------------------------------
def debug_print(*args):
    # Centralize so you can swap to logging later if needed
    print(*args)


def safe_json(resp: requests.Response) -> Any:
    """Try to parse JSON; if it fails, return raw text in a dict."""
    try:
        return resp.json()
    except ValueError:
        return {"content": resp.text}


def check_answer(ai_answer: list[list[int | str]], puzzle_data: dict) -> bool:
    correct_output = puzzle_data.get("test", [])[0].get("output")

    # Convert AI answer cells to ints
    normalized_ai = [[str(cell) for cell in row] for row in ai_answer]

    # Convert correct output cells to ints (in case they’re strings)
    normalized_correct = [[str(cell) for cell in row] for row in correct_output]

    print(f"AI answer: {normalized_ai}")
    print(f"Correct output: {normalized_correct}")

    return normalized_ai == normalized_correct


def build_grid_puzzle_prompt(
    puzzle_data: dict,
    instruct_type: str | None,
) -> str:
    """Build a guidance-first prompt for grid-based puzzles without adding output-format rules."""

    print(f"BUILDING PROMPT FOR {instruct_type}")
    puzzle_hints = puzzle_data["train"]
    puzzle_test_input = puzzle_data["test"][0]["input"]

    # Role + method (concise, but directive)
    prompt = (
        "You are an expert at solving grid-based logic puzzles (ARC-style). "
        "Infer a single, simple, deterministic transformation rule that fits ALL examples, "
        "then apply it to the test input.\n\n"
        "REASONING PROCESS (follow in order):\n"
        "1) Read the GUIDANCE section (if present) and adopt it as your PRIMARY LENS.\n"
        "2) Re-examine each example through that lens; extract a precise rule that fits all.\n"
        "3) Apply the exact rule to the test input step by step.\n"
        "4) Verify the result against constraints (palette, bounds, invariants).\n\n"
    )

    # Guidance section (priority handling)
    if instruct_type == "metaphor":
        metaphor = (puzzle_data.get("metaphor") or "").strip()
        prompt += (
            "### GUIDANCE (METAPHOR) ###\n"
            "Treat this metaphor as the key interpretive frame (objects, motions, interactions). "
            "Your first task is to TRANSLATE the metaphor into explicit, testable, literal grid operations, "
            "and use those to derive a single rule that fits all examples:\n\n"
            "The metaphor you will be basing your work on is:\n"
            f"{metaphor}\n\n"
            "### PHASE 1 — CONCEPTUAL DIGESTION ###\n"
            "Before considering grids or mechanics, reflect deeply on the metaphor itself.\n"
            "You are not analyzing pixels yet—you are understanding *meaning*.\n\n"
            "1) **Cast & Roles** — In plain language, name the key roles or entities implied by the metaphor "
            "(e.g., agent, target, obstacle, boundary, helper).\n"
            "2) **World & Affordances** — Describe what kind of environment or system this metaphor suggests. "
            "What kinds of actions, forces, or relationships exist here?\n"
            "3) **Dynamics** — Tell the causal story: who acts on whom, what changes occur, when the action stops, "
            "and what success looks like.\n"
            "4) **Invariants** — Identify what conceptually remains true throughout the process "
            "(e.g., the actor always chases, the path stays connected, the shape stays whole).\n"
            "5) **Goal Signature** — In one clear sentence, state the end state or desired outcome the metaphor implies.\n\n"
            "➡ Output a 3–6 line **Conceptual Model** summary that captures the metaphor’s logic and intent.\n"
            "Avoid mentioning colors, coordinates, or any grid operations at this stage.\n\n"
        )
    elif instruct_type == "literal":
        literal = (puzzle_data.get("literal_instruction") or "").strip()
        prompt += (
            "### GUIDANCE (LITERAL INSTRUCTIONS) ###\n"
            "Follow these steps strictly as the primary lens for identifying and applying the rule. Instructions:\n"
            f"{literal}\n\n"
        )
    else:
        # RAW mode checklist
        prompt += (
            "### GUIDANCE (RAW MODE) ###\n"
            "- Identify objects and any background/special cells.\n"
            "- Detect objects (contiguous regions), sizes, bounding boxes, adjacency/spacing, alignment.\n"
            "- Seek invariants across examples (counts, symmetry, axes, color mappings, parity).\n"
            "- Test common transforms: copy/move, reflect/rotate, mirror, draw lines/frames, flood-fill, "
            "crop/expand, per-object recolor, nearest alignment, majority/minority rules.\n"
            "- Prefer the simplest rule that explains ALL examples without exceptions.\n\n"
        )

    # Constraints that reduce drift
    prompt += (
        "### CONSTRAINTS ###\n"
        "- Do NOT introduce new objects unless compelled by ALL examples.\n"
        "- Keep output size/bounds unless every example shows a consistent size transform.\n"
        "- If multiple rules fit, choose the simplest fully consistent rule.\n"
        "- Final self-check: object counts, placements, symmetry, and any stated invariants.\n"
        "- In METAPHOR mode: first translate the metaphor into explicit literal operations and internally test them on the examples. "
        "If no consistent mapping exists, IGNORE the metaphor and revert to RAW mode heuristics.\n"
        "- Do not output explanations—only the final grid for the test case.\n\n"
    )

    # Examples
    prompt += "### EXAMPLES ###\n"
    for i, hint in enumerate(puzzle_hints, 1):
        prompt += f"--- Example {i} ---\n"
        prompt += f"INPUT:\n{format_grid(hint['input'])}\n\n"
        prompt += f"OUTPUT:\n{format_grid(hint['output'])}\n\n"

    # Test case (no output-formatting instructions)
    prompt += (
        "### TEST CASE TO SOLVE ###\n"
        "Apply the inferred rule to the test input.\n\n"
        f"INPUT:\n{format_grid(puzzle_test_input)}\n\n"
        "OUTPUT:\n"
    )

    prompt += (
        "IMPORTANT: You must return the final answer via the provided tool as JSON with both keys: "
        "The schema you must output:"
        """"
        "reasoning": {
            "type": "string",
            "description": "Explanation or reasoning behind the prediction. This should be a detailed, chain-of-thought explanation of how the output grid was derived from the input, describing the rules, patterns, or transformations applied.",
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
        """
    )

    # print(prompt)
    return prompt


def call_ai_solver(
    puzzle_data: dict,
    instruct_type: str,
    model: str,
    *,
    provider: str = "openai",  # "ollama" | "openai" | "gemini" | "claude" | "grok"
    temperature: float = 0.1,
    max_tokens: Optional[int] = 64000,
    think: bool = True,
) -> tuple[list[list[int]], str]:
    """
    provider: which adapter to use
    model:    the model name for that provider
    """

    prompt = build_grid_puzzle_prompt(puzzle_data, instruct_type)

    # Use the adapter
    adapter = get_llm_adapter(provider)
    print(f"About to begin with {adapter} and {model}")
    raw_text, reasoning = adapter.generate(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_schema=MARCOUTPUT,
    )

    cleaned_for_json = strip_think(raw_text)
    # print(f"JSON IS is: {cleaned_for_json}")

    parsed_completion = _parse_json_lenient(cleaned_for_json)

    ai_answer = parsed_completion["output_grid"]
    return ai_answer, reasoning
