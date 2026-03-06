import argparse
import asyncio
import os
from typing import Any

from agent.rescue_agent import RescueAgent
from drones.mcp_server import DroneMCPServer
from simulation.spawn import create_simulation


def _scenarios() -> dict[str, dict[str, Any]]:
    return {
        "default": {
            "width": 6,
            "height": 6,
            "drone_count": 3,
            "survivor_count": 5,
            "seed": 7,
            "base": (0, 0),
        },
        "dense_survivors": {
            "width": 6,
            "height": 6,
            "drone_count": 3,
            "survivor_count": 12,
            "seed": 3,
            "base": (0, 0),
        },
        "wide_sparse": {
            "width": 10,
            "height": 8,
            "drone_count": 4,
            "survivor_count": 6,
            "seed": 21,
            "base": (0, 0),
        },
        "low_battery": {
            "width": 6,
            "height": 6,
            "drone_count": 3,
            "survivor_count": 5,
            "seed": 11,
            "base": (0, 0),
            "drone_batteries": [25, 18, 14],
        },
        "edge_base": {
            "width": 8,
            "height": 8,
            "drone_count": 3,
            "survivor_count": 7,
            "seed": 9,
            "base": (0, 7),
        },
        "split_positions": {
            "width": 8,
            "height": 8,
            "drone_count": 3,
            "survivor_count": 7,
            "seed": 5,
            "base": (0, 0),
            "drone_positions": [(0, 0), (7, 0), (7, 7)],
        },
    }


def _apply_scenario(drones, scenario: dict[str, Any]) -> tuple[int, int]:
    base = scenario.get("base", (0, 0))
    for drone in drones:
        drone.base_x = base[0]
        drone.base_y = base[1]
        drone.x = base[0]
        drone.y = base[1]
    positions = scenario.get("drone_positions")
    if positions:
        if len(positions) != len(drones):
            raise ValueError("drone_positions_length_mismatch")
        width = scenario["width"]
        height = scenario["height"]
        for drone, pos in zip(drones, positions, strict=True):
            if not (0 <= pos[0] < width and 0 <= pos[1] < height):
                raise ValueError("drone_position_out_of_bounds")
            drone.x = pos[0]
            drone.y = pos[1]
            drone.status = "idle"
    batteries = scenario.get("drone_batteries")
    if batteries:
        if len(batteries) != len(drones):
            raise ValueError("drone_batteries_length_mismatch")
        for drone, battery in zip(drones, batteries, strict=True):
            drone.battery = battery
    return base


def run_offline_mission(scenario_name: str) -> None:
    scenario = _scenarios().get(scenario_name)
    if scenario is None:
        names = ", ".join(sorted(_scenarios().keys()))
        raise ValueError(f"unknown_scenario: {scenario_name}. available: {names}")
    grid, drones = create_simulation(
        width=scenario["width"],
        height=scenario["height"],
        drone_count=scenario["drone_count"],
        survivor_count=scenario["survivor_count"],
        seed=scenario["seed"],
    )
    base = _apply_scenario(drones, scenario)
    server = DroneMCPServer(grid, drones).server
    model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    agent = RescueAgent(width=grid.width, height=grid.height, base=base, model=model, ollama_url=ollama_url)
    print(f"Scenario: {scenario_name}")
    print("Initial Grid:")
    print(grid.render([d.state() for d in drones]))
    result = asyncio.run(agent.run(server))
    print("Mission Complete")
    print(f"Scanned: {result['scanned']}")
    print(f"Survivors Found: {result['survivors_found']}")
    print("Final Grid:")
    print(grid.render([d.state() for d in drones]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=os.getenv("SCENARIO", "default"))
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()
    if args.list_scenarios:
        print("Available scenarios:")
        for name in sorted(_scenarios().keys()):
            print(f"- {name}")
    else:
        run_offline_mission(args.scenario)
