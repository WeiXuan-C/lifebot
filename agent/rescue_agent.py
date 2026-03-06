import asyncio
import json
import logging
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp.client import Client


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
                result = json.loads(body)
        except (URLError, TimeoutError, json.JSONDecodeError):
            return None
        return result.get("response")


class RescueAgent:
    def __init__(
        self,
        width: int,
        height: int,
        base: tuple[int, int] = (0, 0),
        model: str = "mistral:7b-instruct",
        ollama_url: str = "http://localhost:11434",
        log_path: str = "mission.log",
    ):
        self.width = width
        self.height = height
        self.base = base
        self.unscanned: set[tuple[int, int]] = {
            (x, y) for y in range(height) for x in range(width)
        }
        self.found_survivors: list[tuple[int, int]] = []
        self.known_survivors: dict[tuple[int, int], int] = {}
        self.rescued: set[tuple[int, int]] = set()
        self.scan_cost = 5
        self.reserve_power = 10
        self.llm = OllamaClient(ollama_url, model)
        self.logger = self._build_logger(log_path)

    def _build_logger(self, log_path: str) -> logging.Logger:
        logger = logging.getLogger("lifebot")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    async def run(self, server) -> dict[str, Any]:
        async with Client(server) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools.tools]
            self.logger.info(f"Tools discovered: {', '.join(tool_names)}")
            discovered = await self._list_drones(client)
            self.logger.info(f"Drones discovered: {[d['id'] for d in discovered]}")
            step = 0
            while self.unscanned:
                step += 1
                drones = await self._list_drones(client)
                if not drones:
                    self.logger.info("No active drones, mission aborted")
                    break
                for drone in drones:
                    if not self.unscanned:
                        break
                    await self._act_for_drone(client, drone, tool_names)
                await asyncio.sleep(0)
            return {
                "steps": step,
                "scanned": self.width * self.height - len(self.unscanned),
                "survivors_found": self.found_survivors,
            }

    async def _list_drones(self, client: Client) -> list[dict[str, Any]]:
        result = await client.call_tool("list_drones", {})
        return result.structured_content["drones"]

    async def _act_for_drone(self, client: Client, drone: dict[str, Any], tool_names: list[str]) -> None:
        drone_id = drone["id"]
        battery_status = await client.call_tool("get_battery_status", {"drone_id": drone_id})
        battery = battery_status.structured_content["battery"]
        drone_state = {**drone, "battery": battery}
        plan = self._plan_action(drone_state, tool_names)
        if plan is not None:
            action, arguments = self._normalize_plan(plan, drone_state)
            if action:
                target = None
                if arguments and "x" in arguments and "y" in arguments:
                    target = (arguments["x"], arguments["y"])
                decision = {
                    "drone_id": drone_id,
                    "battery": battery,
                    "position": (drone_state["x"], drone_state["y"]),
                    "action": action,
                    "target": target,
                    "reason": plan.get("reason", "planner"),
                    "unscanned_remaining": len(self.unscanned),
                    "known_survivors": len(self.known_survivors),
                }
                self.logger.info(f"Reasoning: {plan.get('reason', 'planner')} | Decision: {decision}")
                await self._execute_action(client, drone_state, action, arguments)
                return
        await self._act_for_drone_heuristic(client, drone_state)

    async def _act_for_drone_heuristic(self, client: Client, drone: dict[str, Any]) -> None:
        drone_id = drone["id"]
        battery = drone["battery"]
        position = (drone["x"], drone["y"])
        return_cost = self._distance(position, self.base)
        if battery <= return_cost + self.reserve_power:
            rationale = self._rationale(
                drone_id,
                battery,
                position,
                "return_to_base",
                target=self.base,
                reason="battery_guard",
            )
            self.logger.info(rationale)
            await self._follow_path(client, drone_id, self._path_between(position, self.base), scan_on_path=False)
            return
        rescue_target = self._select_rescue_target(drone)
        if rescue_target is not None:
            path = self._path_between(position, rescue_target)
            required = self._path_cost(path) + self.scan_cost + self._distance(rescue_target, self.base)
            if battery < required:
                rationale = self._rationale(
                    drone_id,
                    battery,
                    position,
                    "return_to_base",
                    target=self.base,
                    reason="battery_insufficient_for_rescue",
                )
                self.logger.info(rationale)
                await self._follow_path(client, drone_id, self._path_between(position, self.base), scan_on_path=False)
                return
            rationale = self._rationale(
                drone_id,
                battery,
                position,
                "rescue_target",
                target=rescue_target,
                reason="known_survivor",
            )
            self.logger.info(rationale)
            await self._follow_path(client, drone_id, path, scan_on_path=False)
            scan = await self._call_tool(client, "thermal_scan", {"drone_id": drone_id})
            await self._handle_scan(scan)
            self.rescued.add(rescue_target)
            return
        target = self._select_scan_target(drone)
        if target is None:
            return
        path = self._path_between(position, target)
        scan_count = self._estimate_scan_count(path)
        required = self._path_cost(path) + scan_count * self.scan_cost + self._distance(target, self.base)
        if battery < required:
            rationale = self._rationale(
                drone_id,
                battery,
                position,
                "return_to_base",
                target=self.base,
                reason="battery_insufficient_for_scan",
            )
            self.logger.info(rationale)
            await self._follow_path(client, drone_id, self._path_between(position, self.base), scan_on_path=False)
            return
        rationale = self._rationale(
            drone_id,
            battery,
            position,
            "scan_target",
            target=target,
            reason="coverage_priority",
        )
        self.logger.info(rationale)
        await self._follow_path(client, drone_id, path, scan_on_path=True)

    def _plan_action(self, drone: dict[str, Any], tool_names: list[str]) -> dict[str, Any] | None:
        known_survivors = [
            {"x": pos[0], "y": pos[1], "severity": severity}
            for pos, severity in self.known_survivors.items()
            if pos not in self.rescued
        ]
        unscanned = self._sample_unscanned((drone["x"], drone["y"]))
        state = {
            "grid": {"width": self.width, "height": self.height, "base": self.base},
            "drone": drone,
            "known_survivors": known_survivors,
            "unscanned_candidates": unscanned,
            "tools": tool_names,
            "scan_cost": self.scan_cost,
            "reserve_power": self.reserve_power,
            "rules": [
                "Use only tools listed in tools.",
                "Choose one action from: move_to, move_to_and_scan, thermal_scan, return_to_base, idle.",
                "Prefer scanning unscanned cells; prioritize known survivors by severity and distance.",
                "Keep enough battery to return to base when possible.",
                "Return JSON only.",
            ],
        }
        prompt = (
            "You are the autonomous command agent for an offline rescue swarm. "
            "Decide the next action for the given drone. "
            "Return JSON with keys: action, arguments, reason. "
            "Arguments should include x,y for move_to or move_to_and_scan. "
            "Use a short reason (1-2 sentences) without chain-of-thought.\n"
            f"State: {json.dumps(state, ensure_ascii=False)}\n"
            "Response:"
        )
        response = self.llm.generate(prompt)
        if not response:
            return None
        return self._extract_json(response)

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _normalize_plan(self, plan: dict[str, Any], drone: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        action_raw = str(plan.get("action", "")).strip().lower()
        arguments = plan.get("arguments") or {}
        if action_raw in {"scan", "thermal_scan"}:
            action = "thermal_scan"
        elif action_raw in {"move_to", "move_to_and_scan", "return_to_base", "idle"}:
            action = action_raw
        else:
            return None, {}
        if action == "return_to_base":
            action = "move_to"
            arguments = {"x": self.base[0], "y": self.base[1]}
        if action in {"move_to", "move_to_and_scan"}:
            x = arguments.get("x")
            y = arguments.get("y")
            if not isinstance(x, int) or not isinstance(y, int):
                return None, {}
            if not (0 <= x < self.width and 0 <= y < self.height):
                return None, {}
            move_cost = self._distance((drone["x"], drone["y"]), (x, y))
            if drone["battery"] < move_cost:
                if (x, y) == self.base:
                    return "idle", {}
                return self._normalize_plan({"action": "return_to_base"}, drone)
            if action == "move_to_and_scan":
                if drone["battery"] < move_cost + self.scan_cost:
                    action = "move_to"
            return action, {"x": x, "y": y}
        if action == "thermal_scan":
            if drone["battery"] < self.scan_cost:
                return self._normalize_plan({"action": "return_to_base"}, drone)
            return action, {}
        if action == "idle":
            return action, {}
        return None, {}

    async def _execute_action(
        self,
        client: Client,
        drone: dict[str, Any],
        action: str,
        arguments: dict[str, Any],
    ) -> None:
        drone_id = drone["id"]
        if action == "idle":
            self.logger.info(f"Reasoning: idle | Decision: {{'drone_id': '{drone_id}', 'action': 'idle'}}")
            return
        if action == "thermal_scan":
            scan = await self._call_tool(client, "thermal_scan", {"drone_id": drone_id})
            await self._handle_scan(scan)
            return
        if action == "move_to":
            await self._call_tool(
                client,
                "move_to",
                {"drone_id": drone_id, "x": arguments["x"], "y": arguments["y"]},
            )
            return
        if action == "move_to_and_scan":
            await self._call_tool(
                client,
                "move_to",
                {"drone_id": drone_id, "x": arguments["x"], "y": arguments["y"]},
            )
            if (arguments["x"], arguments["y"]) in self.unscanned:
                scan = await self._call_tool(client, "thermal_scan", {"drone_id": drone_id})
                await self._handle_scan(scan)
            return

    def _sample_unscanned(self, position: tuple[int, int], limit: int = 36) -> list[dict[str, int]]:
        if len(self.unscanned) <= limit:
            return [{"x": pos[0], "y": pos[1]} for pos in self.unscanned]
        ordered = sorted(self.unscanned, key=lambda pos: self._distance(position, pos))
        return [{"x": pos[0], "y": pos[1]} for pos in ordered[:limit]]

    async def _follow_path(
        self,
        client: Client,
        drone_id: str,
        path: list[tuple[int, int]],
        scan_on_path: bool,
    ) -> None:
        if scan_on_path and path and path[0] in self.unscanned:
            scan = await self._call_tool(client, "thermal_scan", {"drone_id": drone_id})
            await self._handle_scan(scan)
        for step in path[1:]:
            await self._call_tool(client, "move_to", {"drone_id": drone_id, "x": step[0], "y": step[1]})
            if scan_on_path and step in self.unscanned:
                scan = await self._call_tool(client, "thermal_scan", {"drone_id": drone_id})
                await self._handle_scan(scan)

    async def _call_tool(self, client: Client, name: str, arguments: dict[str, Any]):
        self.logger.info(f"Action: {name} {arguments}")
        result = await client.call_tool(name, arguments)
        if result.structured_content:
            self.logger.info(f"Result: {name} {result.structured_content}")
        return result

    async def _handle_scan(self, scan_result) -> None:
        content = scan_result.structured_content
        position = (content["x"], content["y"])
        if position in self.unscanned:
            self.unscanned.remove(position)
        if content["found"]:
            self.found_survivors.append(position)
            self.known_survivors[position] = content["severity"] or 1
            self.logger.info(f"Survivor detected at {position} severity={self.known_survivors[position]}")
        else:
            self.logger.info(f"No survivor at {position}")

    def _select_rescue_target(self, drone: dict[str, Any]) -> tuple[int, int] | None:
        candidates = [pos for pos in self.known_survivors.keys() if pos not in self.rescued]
        if not candidates:
            return None
        best_score = None
        best_target = None
        for pos in candidates:
            distance = self._distance((drone["x"], drone["y"]), pos)
            severity = self.known_survivors[pos]
            score = severity * 10 - distance
            if best_score is None or score > best_score:
                best_score = score
                best_target = pos
        return best_target

    def _select_scan_target(self, drone: dict[str, Any]) -> tuple[int, int] | None:
        if not self.unscanned:
            return None
        start = (drone["x"], drone["y"])
        return min(self.unscanned, key=lambda pos: self._distance(start, pos))

    def _estimate_scan_count(self, path: list[tuple[int, int]]) -> int:
        return sum(1 for step in path[1:] if step in self.unscanned)

    def _path_cost(self, path: list[tuple[int, int]]) -> int:
        return max(0, len(path) - 1)

    def _path_between(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = start
        gx, gy = goal
        path = [(x, y)]
        while x != gx or y != gy:
            if x < gx:
                x += 1
            elif x > gx:
                x -= 1
            elif y < gy:
                y += 1
            else:
                y -= 1
            path.append((x, y))
        return path

    def _distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _rationale(
        self,
        drone_id: str,
        battery: int,
        position: tuple[int, int],
        action: str,
        target: tuple[int, int] | None,
        reason: str,
    ) -> str:
        summary = {
            "drone_id": drone_id,
            "battery": battery,
            "position": position,
            "action": action,
            "target": target,
            "reason": reason,
            "unscanned_remaining": len(self.unscanned),
            "known_survivors": len(self.known_survivors),
        }
        prompt = (
            "You are an offline rescue mission planner. Provide a brief decision rationale "
            "in 2-3 sentences, focusing on battery, distance, coverage, and survivor priority. "
            "Do not include step-by-step chain-of-thought or calculations.\n"
            f"State: {json.dumps(summary, ensure_ascii=False)}\n"
            "Rationale:"
        )
        response = self.llm.generate(prompt)
        if response:
            return f"Reasoning: {response.strip()} | Decision: {summary}"
        return f"Reasoning: decision derived from mission heuristics | Decision: {summary}"


async def run_mission(server, width: int, height: int, log_path: str) -> dict[str, Any]:
    agent = RescueAgent(width=width, height=height, log_path=log_path)
    return await agent.run(server)


def run_mission_blocking(server, width: int, height: int, log_path: str) -> dict[str, Any]:
    return asyncio.run(run_mission(server, width, height, log_path))
