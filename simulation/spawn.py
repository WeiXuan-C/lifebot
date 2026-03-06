import random

from drones.drone import Drone
from simulation.grid import Grid


def create_simulation(
    width: int,
    height: int,
    drone_count: int,
    survivor_count: int,
    seed: int = 42,
) -> tuple[Grid, list[Drone]]:
    random.seed(seed)
    grid = Grid(width, height)
    all_positions = grid.all_positions()
    random.shuffle(all_positions)
    for x, y in all_positions[:survivor_count]:
        grid.set_survivor(x, y)
    drones: list[Drone] = []
    for i in range(drone_count):
        drones.append(Drone(drone_id=f"drone_{i+1}", x=0, y=0))
    return grid, drones
