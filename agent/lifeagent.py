import asyncio
from typing import Any

from mcp.client import Client


class LifeBotAgent:
    def __init__(self, width: int, height: int, base: tuple[int, int] = (0, 0)):
        self.width = width
        self.height = height
        self.base = base
        self.unscanned: set[tuple[int, int]] = {
            (x, y) for y in range(height) for x in range(width)
        }
        self.found_survivors: list[tuple[int, int]] = []
        self.battery_low = 40

    async def run(self, server) -> dict[str, Any]:
        async with Client(server) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"发现工具: {', '.join(tool_names)}")
            discovered = await self._list_drones(client)
            print(f"发现无人机: {[d['id'] for d in discovered]}")
            step = 0
            while self.unscanned:
                step += 1
                drones = await self._list_drones(client)
                if not drones:
                    print("未发现可用无人机，任务中止")
                    break
                for drone in drones:
                    if not self.unscanned:
                        break
                    await self._act_for_drone(client, drone)
                await asyncio.sleep(0)
            return {
                "steps": step,
                "scanned": self.width * self.height - len(self.unscanned),
                "survivors_found": self.found_survivors,
            }

    async def _list_drones(self, client: Client) -> list[dict[str, Any]]:
        result = await client.call_tool("list_drones", {})
        return result.structured_content["drones"]

    async def _act_for_drone(self, client: Client, drone: dict[str, Any]) -> None:
        drone_id = drone["id"]
        battery_status = await client.call_tool("get_battery_status", {"drone_id": drone_id})
        battery = battery_status.structured_content["battery"]
        position = (drone["x"], drone["y"])
        if battery <= self.battery_low:
            if position != self.base:
                print(f"推理: {drone_id} 电量低({battery})，返回基地充电")
                await client.call_tool("move_to", {"drone_id": drone_id, "x": self.base[0], "y": self.base[1]})
            else:
                print(f"推理: {drone_id} 在基地补电")
                await client.call_tool("move_to", {"drone_id": drone_id, "x": self.base[0], "y": self.base[1]})
            return
        target = self._select_target(drone)
        if target is None:
            return
        print(f"推理: {drone_id} 距离 {target} 最近，分配扫描")
        await client.call_tool("move_to", {"drone_id": drone_id, "x": target[0], "y": target[1]})
        scan = await client.call_tool("thermal_scan", {"drone_id": drone_id})
        scanned_pos = (scan.structured_content["x"], scan.structured_content["y"])
        if scanned_pos in self.unscanned:
            self.unscanned.remove(scanned_pos)
        if scan.structured_content["found"]:
            self.found_survivors.append(scanned_pos)
            print(f"发现幸存者: {scanned_pos}")

    def _select_target(self, drone: dict[str, Any]) -> tuple[int, int] | None:
        if not self.unscanned:
            return None
        dx = drone["x"]
        dy = drone["y"]
        return min(self.unscanned, key=lambda p: abs(p[0] - dx) + abs(p[1] - dy))
