from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ScenarioEvent:
    step: int
    description: str
    inputs: Dict[str, Any] | None = None
    action: str | None = None


@dataclass
class Scenario:
    name: str
    description: str
    initial_actions: List[str]
    expected_final_state: str
    expected_alert_level: str
    events: List[ScenarioEvent]


class ScenarioLoader:
    def __init__(self, scenario_dir: str = "scenarios") -> None:
        self.scenario_dir = Path(scenario_dir)

    def load(self, filename: str) -> Scenario:
        path = self.scenario_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        events = [
            ScenarioEvent(
                step=event["step"],
                description=event["description"],
                inputs=event.get("inputs"),
                action=event.get("action"),
            )
            for event in raw.get("events", [])
        ]
        return Scenario(
            name=raw["name"],
            description=raw["description"],
            initial_actions=raw.get("initial_actions", []),
            expected_final_state=raw["expected_final_state"],
            expected_alert_level=raw["expected_alert_level"],
            events=events,
        )

    def list_scenarios(self) -> List[str]:
        if not self.scenario_dir.exists():
            return []
        return sorted([p.name for p in self.scenario_dir.glob("*.json")])