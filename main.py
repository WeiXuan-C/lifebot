import asyncio

from agent.lifeagent import LifeBotAgent
from drones.mcp_server import DroneMCPServer
from simulation.spawn import create_simulation


async def run_demo() -> None:
    grid, drones = create_simulation(width=6, height=6, drone_count=3, survivor_count=5, seed=7)
    server = DroneMCPServer(grid, drones).server
    agent = LifeBotAgent(width=grid.width, height=grid.height, base=(0, 0))
    print("初始网格:")
    print(grid.render([d.state() for d in drones]))
    result = await agent.run(server)
    print("任务完成")
    print(f"扫描总数: {result['scanned']}")
    print(f"发现幸存者: {result['survivors_found']}")
    print("最终网格:")
    print(grid.render([d.state() for d in drones]))


if __name__ == "__main__":
    asyncio.run(run_demo())
