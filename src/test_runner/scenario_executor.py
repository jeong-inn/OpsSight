from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.test_runner.scenario_loader import Scenario
from src.core.runtime_controller import RuntimeController


@dataclass
class ScenarioExecutionResult:
    scenario_name: str
    final_state: str
    final_alert_level: str
    last_event: str | None
    recommended_action: str
    step_results: List[Dict[str, Any]]


class ScenarioExecutor:
    def run(self, scenario: Scenario) -> ScenarioExecutionResult:
        controller = RuntimeController()

        for action in scenario.initial_actions:
            self._execute_action(controller, action)

        step_results: List[Dict[str, Any]] = []
        latest_snapshot = None

        for event in scenario.events:
            if event.action:
                self._execute_action(controller, event.action)

                latest_snapshot = {
                    "current_state": controller.state_machine.get_state(),
                    "alert_level": "INFO",
                    "alert_score": 0.0,
                    "reasons": [f"manual action executed: {event.action}"],
                    "recommended_action": "No immediate action required.",
                    "last_event": controller.last_event,
                }

                step_results.append(
                    {
                        "step": event.step,
                        "description": event.description,
                        "action": event.action,
                        "state": latest_snapshot["current_state"],
                        "alert_level": latest_snapshot["alert_level"],
                        "alert_score": latest_snapshot["alert_score"],
                        "last_event": latest_snapshot["last_event"],
                        "recommended_action": latest_snapshot["recommended_action"],
                        "reasons": latest_snapshot["reasons"],
                    }
                )

            elif event.inputs:
                snapshot = controller.evaluate_and_update(**event.inputs)
                latest_snapshot = snapshot

                step_results.append(
                    {
                        "step": event.step,
                        "description": event.description,
                        "action": None,
                        "state": snapshot.current_state,
                        "alert_level": snapshot.alert_level,
                        "alert_score": snapshot.alert_score,
                        "last_event": snapshot.last_event,
                        "recommended_action": snapshot.recommended_action,
                        "reasons": snapshot.reasons,
                    }
                )
            else:
                raise ValueError(
                    f"Scenario event at step {event.step} must contain either 'inputs' or 'action'."
                )

        if latest_snapshot is None:
            raise ValueError("Scenario contains no executable events.")

        if isinstance(latest_snapshot, dict):
            return ScenarioExecutionResult(
                scenario_name=scenario.name,
                final_state=latest_snapshot["current_state"],
                final_alert_level=latest_snapshot["alert_level"],
                last_event=latest_snapshot["last_event"],
                recommended_action=latest_snapshot["recommended_action"],
                step_results=step_results,
            )

        return ScenarioExecutionResult(
            scenario_name=scenario.name,
            final_state=latest_snapshot.current_state,
            final_alert_level=latest_snapshot.alert_level,
            last_event=latest_snapshot.last_event,
            recommended_action=latest_snapshot.recommended_action,
            step_results=step_results,
        )

    def _execute_action(self, controller: RuntimeController, action: str) -> None:
        if action == "boot":
            controller.boot()
        elif action == "arm":
            controller.arm()
        elif action == "activate":
            controller.activate()
        elif action == "reset":
            controller.reset()
        elif action == "stop":
            controller.stop()
        else:
            raise ValueError(f"Unsupported action: {action}")