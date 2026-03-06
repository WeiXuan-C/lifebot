from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from drones.drone import Drone
from simulation.grid import Grid


class DroneMCPServer:
    def __init__(self, grid: Grid, drones: list[Drone], on_update: Callable[[Grid, list[Drone]], None] | None = None):
        self.grid = grid
        self.drones: dict[str, Drone] = {d.drone_id: d for d in drones}
        self.on_update = on_update
        self.server = MCPServer(
            name="lifebot-drone-sim",
            title="LifeBot Drone Simulation",
            description="Offline drone simulation with MCP tools",
        )
        self._register_tools()

    def _register_tools(self) -> None:
        @self.server.tool(name="list_drones", structured_output=True)
        def list_drones() -> dict[str, Any]:
            return {"drones": [d.state() for d in self.drones.values()]}

        @self.server.tool(name="move_to", structured_output=True)
        def move_to(drone_id: str, x: int, y: int) -> dict[str, Any]:
            drone = self._get_drone(drone_id)
            state = drone.move_to(x, y, self.grid)
            self._notify_update()
            return {"drone": state}

        @self.server.tool(name="get_battery_status", structured_output=True)
        def get_battery_status(drone_id: str) -> dict[str, Any]:
            drone = self._get_drone(drone_id)
            return {"drone_id": drone.drone_id, "battery": drone.battery, "status": drone.status}

        @self.server.tool(name="thermal_scan", structured_output=True)
        def thermal_scan(drone_id: str) -> dict[str, Any]:
            drone = self._get_drone(drone_id)
            result = drone.thermal_scan(self.grid)
            result["drone_id"] = drone.drone_id
            self._notify_update()
            return result

    def _notify_update(self) -> None:
        if self.on_update:
            self.on_update(self.grid, list(self.drones.values()))

    def _get_drone(self, drone_id: str) -> Drone:
        if drone_id not in self.drones:
            raise ValueError("unknown_drone")
        return self.drones[drone_id]
