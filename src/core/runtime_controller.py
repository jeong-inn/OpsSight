from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from src.core.fault_tracker import FaultTracker
from src.core.state_machine import (
    OperationalStateMachine,
    SystemEvent,
    SystemState,
)
from src.core.alert_engine import AlertEngine, AlertLevel, AlertResult


@dataclass
class RuntimeSnapshot:
    current_state: str
    alert_level: str
    alert_score: float
    reasons: list[str]
    recommended_action: str
    last_event: Optional[str]
    transition_allowed: bool
    transition_changed: bool
    warning_streak: int
    critical_streak: int
    normal_streak: int
    persistent_critical: bool


class RuntimeController:
    def __init__(self) -> None:
        self.state_machine = OperationalStateMachine()
        self.alert_engine = AlertEngine()
        self.fault_tracker = FaultTracker()
        self.last_event: Optional[str] = None

    def boot(self) -> None:
        result = self.state_machine.trigger(
            SystemEvent.BOOT_COMPLETE,
            reason="system boot completed"
        )
        self.last_event = result.event.value

    def arm(self) -> None:
        result = self.state_machine.trigger(
            SystemEvent.START_COMMAND,
            reason="operator armed the system"
        )
        self.last_event = result.event.value

    def activate(self) -> None:
        result = self.state_machine.trigger(
            SystemEvent.START_COMMAND,
            reason="operator activated the system"
        )
        self.last_event = result.event.value

    def reset(self) -> None:
        result = self.state_machine.trigger(
            SystemEvent.RESET_COMMAND,
            reason="manual reset requested"
        )
        self.fault_tracker.reset()
        self.last_event = result.event.value

    def stop(self) -> None:
        result = self.state_machine.trigger(
            SystemEvent.STOP_COMMAND,
            reason="operator stop requested"
        )
        self.last_event = result.event.value

    def evaluate_and_update(
        self,
        anomaly_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        anomaly_count: int = 0,
        persistent_fault: bool = False,
        communication_delay: bool = False,
        sensor_stuck: bool = False,
        compound_fault: bool = False,
        recovery_confirmed: bool = False,
    ) -> RuntimeSnapshot:
        alert_result: AlertResult = self.alert_engine.evaluate(
            anomaly_score=anomaly_score,
            risk_score=risk_score,
            anomaly_count=anomaly_count,
            persistent_fault=persistent_fault,
            communication_delay=communication_delay,
            sensor_stuck=sensor_stuck,
            compound_fault=compound_fault,
        )

        tracker_snapshot = self.fault_tracker.update(alert_result.level.value)

        effective_persistent_fault = (
            persistent_fault or tracker_snapshot.persistent_critical
        )

        effective_recovery_confirmed = (
            recovery_confirmed and tracker_snapshot.normal_streak >= 1
        )

        event = self._map_alert_to_event(
            alert_level=alert_result.level,
            persistent_fault=effective_persistent_fault,
            recovery_confirmed=effective_recovery_confirmed,
        )

        allowed = False
        changed = False

        if event is not None:
            transition = self.state_machine.trigger(
                event,
                reason="runtime controller decision"
            )
            self.last_event = transition.event.value
            allowed = transition.allowed
            changed = transition.changed
        return RuntimeSnapshot(
            current_state=self.state_machine.get_state(),
            alert_level=alert_result.level.value,
            alert_score=alert_result.score,
            reasons=alert_result.reasons,
            recommended_action=alert_result.recommended_action,
            last_event=self.last_event,
            transition_allowed=allowed,
            transition_changed=changed,
            warning_streak=tracker_snapshot.warning_streak,
            critical_streak=tracker_snapshot.critical_streak,
            normal_streak=tracker_snapshot.normal_streak,
            persistent_critical=tracker_snapshot.persistent_critical,
        )

    def get_history(self) -> list[dict[str, Any]]:
        return self.state_machine.get_history()

    def _map_alert_to_event(
        self,
        alert_level: AlertLevel,
        persistent_fault: bool = False,
        recovery_confirmed: bool = False,
    ) -> Optional[SystemEvent]:
        current = self.state_machine.current_state

        if recovery_confirmed and current == SystemState.DEGRADED:
            return SystemEvent.RECOVERY_CONFIRMED

        if alert_level == AlertLevel.CRITICAL:
            if persistent_fault:
                return SystemEvent.PERSISTENT_FAULT
            return SystemEvent.CRITICAL_FAULT

        if alert_level == AlertLevel.WARNING:
            return SystemEvent.WARNING_FAULT

        return None