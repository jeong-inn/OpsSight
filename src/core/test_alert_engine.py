from src.core.alert_engine import AlertEngine


def main():
    engine = AlertEngine()

    cases = [
        {
            "name": "normal",
            "kwargs": {"anomaly_score": 0.12, "risk_score": 0.18, "anomaly_count": 0},
        },
        {
            "name": "caution_case",
            "kwargs": {"anomaly_score": 0.32, "risk_score": 0.35, "anomaly_count": 1},
        },
        {
            "name": "warning_case",
            "kwargs": {"anomaly_score": 0.58, "risk_score": 0.66, "anomaly_count": 3},
        },
        {
            "name": "critical_case",
            "kwargs": {
                "anomaly_score": 0.72,
                "risk_score": 0.79,
                "anomaly_count": 6,
                "persistent_fault": True,
                "compound_fault": True,
            },
        },
    ]

    for case in cases:
        result = engine.evaluate(**case["kwargs"])
        print(f"\n[{case['name']}]")
        print(f"level={result.level.value}")
        print(f"score={result.score:.2f}")
        print(f"reasons={result.reasons}")
        print(f"action={result.recommended_action}")


if __name__ == "__main__":
    main()