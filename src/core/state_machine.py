from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SystemState(str, Enum):
    INIT = "INIT"
    STANDBY = "STANDBY"
    READY = "READY"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SAFE_SHUTDOWN = "SAFE_SHUTDOWN"
    FAULT = "FAULT"


class SystemEvent(str, Enum):
    BOOT_COMPLETE = "BOOT_COMPLETE"
    START_COMMAND = "START_COMMAND"
    STOP_COMMAND = "STOP_COMMAND"
    RESET_COMMAND = "RESET_COMMAND"
    WARNING_FAULT = "WARNING_FAULT"
    CRITICAL_FAULT = "CRITICAL_FAULT"
    PERSISTENT_FAULT = "PERSISTENT_FAULT"
    RECOVERY_CONFIRMED = "RECOVERY_CONFIRMED"


@dataclass
class TransitionRecord:
    timestamp: str
    previous_state: str
    event: str
    next_state: str
    reason: Optional[str] = None


@dataclass
class StateMachineResult:
    previous_state: SystemState
    current_state: SystemState
    event: SystemEvent
    changed: bool
    allowed: bool
    reason: Optional[str] = None


@dataclass
class OperationalStateMachine:
    current_state: SystemState = SystemState.INIT
    history: List[TransitionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transitions: Dict[Tuple[SystemState, SystemEvent], SystemState] = {
            (SystemState.INIT, SystemEvent.BOOT_COMPLETE): SystemState.STANDBY,

            (SystemState.STANDBY, SystemEvent.START_COMMAND): SystemState.READY,

            (SystemState.READY, SystemEvent.START_COMMAND): SystemState.ACTIVE,
            (SystemState.READY, SystemEvent.STOP_COMMAND): SystemState.STANDBY,

            (SystemState.ACTIVE, SystemEvent.WARNING_FAULT): SystemState.DEGRADED,
            (SystemState.ACTIVE, SystemEvent.CRITICAL_FAULT): SystemState.SAFE_SHUTDOWN,
            (SystemState.ACTIVE, SystemEvent.PERSISTENT_FAULT): SystemState.FAULT,
            (SystemState.ACTIVE, SystemEvent.STOP_COMMAND): SystemState.STANDBY,

            (SystemState.DEGRADED, SystemEvent.RECOVERY_CONFIRMED): SystemState.READY,
            (SystemState.DEGRADED, SystemEvent.CRITICAL_FAULT): SystemState.SAFE_SHUTDOWN,
            (SystemState.DEGRADED, SystemEvent.PERSISTENT_FAULT): SystemState.FAULT,
            (SystemState.DEGRADED, SystemEvent.STOP_COMMAND): SystemState.STANDBY,

            (SystemState.SAFE_SHUTDOWN, SystemEvent.RESET_COMMAND): SystemState.STANDBY,
            (SystemState.FAULT, SystemEvent.RESET_COMMAND): SystemState.STANDBY,
        }

    def trigger(self, event: SystemEvent, reason: Optional[str] = None) -> StateMachineResult:
        previous_state = self.current_state
        key = (self.current_state, event)

        if key not in self.transitions:
            return StateMachineResult(
                previous_state=previous_state,
                current_state=self.current_state,
                event=event,
                changed=False,
                allowed=False,
                reason=reason or f"Transition not allowed from {self.current_state} with event {event}",
            )

        next_state = self.transitions[key]
        self.current_state = next_state

        self.history.append(
            TransitionRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                previous_state=previous_state.value,
                event=event.value,
                next_state=next_state.value,
                reason=reason,
            )
        )

        return StateMachineResult(
            previous_state=previous_state,
            current_state=next_state,
            event=event,
            changed=(previous_state != next_state),
            allowed=True,
            reason=reason,
        )

    def get_state(self) -> str:
        return self.current_state.value

    def get_history(self) -> List[dict]:
        return [
            {
                "timestamp": item.timestamp,
                "previous_state": item.previous_state,
                "event": item.event,
                "next_state": item.next_state,
                "reason": item.reason,
            }
            for item in self.history
        ]

    def reset(self) -> None:
        self.current_state = SystemState.INIT
        self.history.clear()