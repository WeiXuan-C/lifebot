from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str


@dataclass
class ToolList:
    tools: list[ToolDefinition]


@dataclass
class ToolResult:
    structured_content: dict[str, Any] | None


class MCPServer:
    def __init__(self, name: str, title: str, description: str):
        self.name = name
        self.title = title
        self.description = description
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self._definitions: list[ToolDefinition] = []

    def tool(self, name: str, structured_output: bool = False):
        def decorator(func: Callable[..., dict[str, Any]]):
            self._tools[name] = func
            self._definitions.append(ToolDefinition(name=name))
            return func
        return decorator

    def list_tools(self) -> ToolList:
        return ToolList(self._definitions)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            raise ValueError("unknown_tool")
        result = self._tools[name](**arguments)
        return ToolResult(structured_content=result)
