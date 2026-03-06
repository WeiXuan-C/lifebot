from dataclasses import dataclass
from typing import Any

from simulation.grid import Grid


@dataclass
class Drone:
    drone_id: str
    x: int
    y: int
    battery: int = 100
    status: str = "idle"
    base_x: int = 0
    base_y: int = 0

    def state(self) -> dict[str, Any]:
        return {
            "id": self.drone_id,
            "x": self.x,
            "y": self.y,
            "battery": self.battery,
            "status": self.status,
        }

    def distance_to(self, x: int, y: int) -> int:
        return abs(self.x - x) + abs(self.y - y)

    def move_to(self, x: int, y: int, grid: Grid) -> dict[str, Any]:
        if not grid.in_bounds(x, y):
            raise ValueError("target_out_of_bounds")
        cost = self.distance_to(x, y)
        if self.battery < cost:
            raise ValueError("battery_insufficient")
        self.status = "moving"
        self.battery -= cost
        self.x = x
        self.y = y
        if self.x == self.base_x and self.y == self.base_y:
            self.status = "charging"
            self.battery = 100
        else:
            self.status = "idle"
        return self.state()

    def thermal_scan(self, grid: Grid) -> dict[str, Any]:
        cost = 5
        if self.battery < cost:
            raise ValueError("battery_insufficient")
        self.status = "scanning"
        self.battery -= cost
        grid.mark_scanned(self.x, self.y)
        found = grid.has_survivor(self.x, self.y)
        self.status = "idle"
        return {"x": self.x, "y": self.y, "found": found, "battery": self.battery}
