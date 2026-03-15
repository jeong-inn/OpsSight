from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FaultTrackerSnapshot:
    warning_streak: int
    critical_streak: int
    normal_streak: int
    persistent_warning: bool
    persistent_critical: bool


class FaultTracker:
    def __init__(
        self,
        warning_persistence_threshold: int = 2,
        critical_persistence_threshold: int = 2,
    ) -> None:
        self.warning_persistence_threshold = warning_persistence_threshold
        self.critical_persistence_threshold = critical_persistence_threshold

        self.warning_streak = 0
        self.critical_streak = 0
        self.normal_streak = 0

    def update(self, alert_level: str) -> FaultTrackerSnapshot:
        if alert_level == "CRITICAL":
            self.critical_streak += 1
            self.warning_streak = 0
            self.normal_streak = 0

        elif alert_level == "WARNING":
            self.warning_streak += 1
            self.critical_streak = 0
            self.normal_streak = 0

        elif alert_level in ("INFO", "CAUTION"):
            self.normal_streak += 1
            self.warning_streak = 0
            self.critical_streak = 0

        return self.get_snapshot()

    def reset(self) -> None:
        self.warning_streak = 0
        self.critical_streak = 0
        self.normal_streak = 0

    def get_snapshot(self) -> FaultTrackerSnapshot:
        return FaultTrackerSnapshot(
            warning_streak=self.warning_streak,
            critical_streak=self.critical_streak,
            normal_streak=self.normal_streak,
            persistent_warning=self.warning_streak >= self.warning_persistence_threshold,
            persistent_critical=self.critical_streak >= self.critical_persistence_threshold,
        )

    def to_dict(self) -> Dict[str, int | bool]:
        snapshot = self.get_snapshot()
        return {
            "warning_streak": snapshot.warning_streak,
            "critical_streak": snapshot.critical_streak,
            "normal_streak": snapshot.normal_streak,
            "persistent_warning": snapshot.persistent_warning,
            "persistent_critical": snapshot.persistent_critical,
        }