from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class AlertLevel(str, Enum):
    INFO = "INFO"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertResult:
    level: AlertLevel
    score: float
    reasons: List[str]
    recommended_action: str


class AlertEngine:
    def __init__(
        self,
        warning_score_threshold: float = 0.6,
        critical_score_threshold: float = 0.8,
        anomaly_count_warning: int = 3,
        anomaly_count_critical: int = 5,
    ) -> None:
        self.warning_score_threshold = warning_score_threshold
        self.critical_score_threshold = critical_score_threshold
        self.anomaly_count_warning = anomaly_count_warning
        self.anomaly_count_critical = anomaly_count_critical

    def evaluate(
        self,
        anomaly_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        anomaly_count: int = 0,
        persistent_fault: bool = False,
        communication_delay: bool = False,
        sensor_stuck: bool = False,
        compound_fault: bool = False,
    ) -> AlertResult:
        reasons: List[str] = []
        score_candidates: List[float] = []

        if anomaly_score is not None:
            score_candidates.append(float(anomaly_score))
        if risk_score is not None:
            score_candidates.append(float(risk_score))

        final_score = max(score_candidates) if score_candidates else 0.0

        if persistent_fault:
            reasons.append("persistent fault detected")
        if communication_delay:
            reasons.append("communication delay detected")
        if sensor_stuck:
            reasons.append("sensor stuck detected")
        if compound_fault:
            reasons.append("compound fault detected")
        if anomaly_count > 0:
            reasons.append(f"{anomaly_count} anomalous signals observed")

        if compound_fault or persistent_fault or sensor_stuck:
            if final_score < self.critical_score_threshold:
                final_score = self.critical_score_threshold

        if (
            compound_fault
            or persistent_fault
            or sensor_stuck
            or anomaly_count >= self.anomaly_count_critical
            or final_score >= self.critical_score_threshold
        ):
            return AlertResult(
                level=AlertLevel.CRITICAL,
                score=final_score,
                reasons=reasons or ["critical abnormal condition"],
                recommended_action="Trigger safe shutdown and perform immediate inspection.",
            )

        if (
            communication_delay
            or anomaly_count >= self.anomaly_count_warning
            or final_score >= self.warning_score_threshold
        ):
            return AlertResult(
                level=AlertLevel.WARNING,
                score=final_score,
                reasons=reasons or ["warning-level abnormal condition"],
                recommended_action="Switch to degraded operation and inspect affected subsystem.",
            )

        if anomaly_count > 0 or final_score > 0.3:
            return AlertResult(
                level=AlertLevel.CAUTION,
                score=final_score,
                reasons=reasons or ["minor abnormal signal detected"],
                recommended_action="Continue monitoring and review recent signal changes.",
            )

        return AlertResult(
            level=AlertLevel.INFO,
            score=final_score,
            reasons=reasons or ["system operating within normal range"],
            recommended_action="No immediate action required.",
        )