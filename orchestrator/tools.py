"""Tool decorator + registry.

Tools are plain Python functions decorated with @tool. The decorator captures
the function signature into a JSON Schema that we hand to the LLM. Agents that
support tool use receive the schema; the orchestrator dispatches the call when
the LLM emits a tool_use block.

Design note: we keep this provider-agnostic. The agents that actually invoke
tools convert this internal Tool spec into the provider's specific tool format.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema
    fn: Callable

    def __call__(self, **kwargs):
        return self.fn(**kwargs)

    def to_anthropic_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def to_openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def tool(fn: Callable | None = None, *, description: str | None = None) -> Callable:
    """Decorator. Captures docstring as description, type hints as schema."""

    def wrap(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            t = hints.get(name, str)
            properties[name] = {"type": _PY_TO_JSON.get(t, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return Tool(
            name=fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            parameters={"type": "object", "properties": properties, "required": required},
            fn=fn,
        )

    if fn is not None and callable(fn):
        return wrap(fn)
    return wrap


class ToolRegistry:
    """Bind multiple tools to an agent. Look up by name at dispatch time."""

    def __init__(self, tools: list[Tool] | None = None):
        self._by_name: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if not isinstance(t, Tool):
            raise TypeError(f"expected Tool, got {type(t)}; did you forget the @tool decorator?")
        self._by_name[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._by_name:
            raise KeyError(f"unknown tool: {name!r}. available: {list(self._by_name)}")
        return self._by_name[name]

    def names(self) -> list[str]:
        return list(self._by_name)

    def all(self) -> list[Tool]:
        return list(self._by_name.values())


# A couple of example tools the demo uses.

@tool(description="Search the web for a query. Returns a list of {title, url, snippet} dicts.")
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Toy implementation — replace with Brave / Tavily / Exa in production."""
    return [
        {"title": f"Result {i} for {query}", "url": f"https://example.com/{i}", "snippet": f"Snippet about {query}"}
        for i in range(1, max_results + 1)
    ]