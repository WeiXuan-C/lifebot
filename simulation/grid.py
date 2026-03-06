from dataclasses import dataclass
from typing import Iterable


@dataclass
class Cell:
    has_survivor: bool = False
    scanned: bool = False
    survivor_severity: int | None = None


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells = [[Cell() for _ in range(width)] for _ in range(height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> Cell:
        return self._cells[y][x]

    def set_survivor(self, x: int, y: int, severity: int) -> None:
        cell = self.get_cell(x, y)
        cell.has_survivor = True
        cell.survivor_severity = severity

    def mark_scanned(self, x: int, y: int) -> None:
        self.get_cell(x, y).scanned = True

    def is_scanned(self, x: int, y: int) -> bool:
        return self.get_cell(x, y).scanned

    def has_survivor(self, x: int, y: int) -> bool:
        return self.get_cell(x, y).has_survivor

    def survivor_severity(self, x: int, y: int) -> int | None:
        return self.get_cell(x, y).survivor_severity

    def all_positions(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width)]

    def unscanned_positions(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width) if not self.is_scanned(x, y)]

    def survivors_remaining(self) -> int:
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                cell = self.get_cell(x, y)
                if cell.has_survivor and not cell.scanned:
                    count += 1
        return count

    def render(self, drones: Iterable[dict] | None = None) -> str:
        drone_positions: dict[tuple[int, int], str] = {}
        if drones:
            for d in drones:
                drone_positions[(d["x"], d["y"])] = "🛸"
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in drone_positions:
                    row.append(drone_positions[(x, y)])
                    continue
                cell = self.get_cell(x, y)
                if not cell.scanned:
                    row.append("□")
                elif cell.has_survivor:
                    row.append("🧍")
                else:
                    row.append("·")
            lines.append(" ".join(row))
        return "\n".join(lines)
