from src.core.runtime_controller import RuntimeController


def print_snapshot(title, snapshot):
    print(f"\n[{title}]")
    print(f"state={snapshot.current_state}")
    print(f"alert={snapshot.alert_level}")
    print(f"score={snapshot.alert_score:.2f}")
    print(f"reasons={snapshot.reasons}")
    print(f"action={snapshot.recommended_action}")
    print(f"last_event={snapshot.last_event}")
    print(f"allowed={snapshot.transition_allowed}")
    print(f"changed={snapshot.transition_changed}")


def main():
    controller = RuntimeController()

    controller.boot()
    controller.arm()
    controller.activate()

    s1 = controller.evaluate_and_update(
        anomaly_score=0.34,
        risk_score=0.38,
        anomaly_count=1,
    )
    print_snapshot("caution", s1)

    s2 = controller.evaluate_and_update(
        anomaly_score=0.62,
        risk_score=0.68,
        anomaly_count=3,
        communication_delay=True,
    )
    print_snapshot("warning", s2)

    s3 = controller.evaluate_and_update(
        anomaly_score=0.78,
        risk_score=0.83,
        anomaly_count=6,
        persistent_fault=True,
        compound_fault=True,
    )
    print_snapshot("critical", s3)

    controller.reset()

    print("\nHistory:")
    for row in controller.get_history():
        print(row)


if __name__ == "__main__":
    main()