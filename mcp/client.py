from typing import Any


class Client:
    def __init__(self, server):
        self.server = server

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        return self.server.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        return self.server.call_tool(name, arguments)
