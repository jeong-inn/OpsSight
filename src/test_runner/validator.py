from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ValidationResult:
    passed: bool
    expected_final_state: str
    actual_final_state: str
    expected_alert_level: str
    actual_alert_level: str
    details: Dict[str, Any]


class ScenarioValidator:
    def validate(
        self,
        expected_final_state: str,
        actual_final_state: str,
        expected_alert_level: str,
        actual_alert_level: str,
    ) -> ValidationResult:
        state_match = expected_final_state == actual_final_state
        alert_match = expected_alert_level == actual_alert_level
        passed = state_match and alert_match

        return ValidationResult(
            passed=passed,
            expected_final_state=expected_final_state,
            actual_final_state=actual_final_state,
            expected_alert_level=expected_alert_level,
            actual_alert_level=actual_alert_level,
            details={
                "state_match": state_match,
                "alert_match": alert_match,
            },
        )