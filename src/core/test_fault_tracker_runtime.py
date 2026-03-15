from src.core.runtime_controller import RuntimeController


def print_snapshot(title, snapshot):
    print(f"\n[{title}]")
    print("state =", snapshot.current_state)
    print("alert =", snapshot.alert_level)
    print("score =", round(snapshot.alert_score, 2))
    print("last_event =", snapshot.last_event)
    print("warning_streak =", snapshot.warning_streak)
    print("critical_streak =", snapshot.critical_streak)
    print("normal_streak =", snapshot.normal_streak)
    print("persistent_critical =", snapshot.persistent_critical)
    print("reasons =", snapshot.reasons)
    print("action =", snapshot.recommended_action)


def main():
    controller = RuntimeController()
    controller.boot()
    controller.arm()
    controller.activate()

    s1 = controller.evaluate_and_update(
        anomaly_score=0.58,
        risk_score=0.63,
        anomaly_count=3,
        communication_delay=True,
    )
    print_snapshot("warning_1", s1)

    s2 = controller.evaluate_and_update(
        anomaly_score=0.61,
        risk_score=0.66,
        anomaly_count=3,
        communication_delay=True,
    )
    print_snapshot("warning_2", s2)

    s3 = controller.evaluate_and_update(
        anomaly_score=0.79,
        risk_score=0.84,
        anomaly_count=5,
        compound_fault=True,
    )
    print_snapshot("critical_1", s3)

    s4 = controller.evaluate_and_update(
        anomaly_score=0.81,
        risk_score=0.86,
        anomaly_count=5,
        compound_fault=True,
    )
    print_snapshot("critical_2", s4)

    s5 = controller.evaluate_and_update(
        anomaly_score=0.12,
        risk_score=0.18,
        anomaly_count=0,
        recovery_confirmed=True,
    )
    print_snapshot("recovery", s5)


if __name__ == "__main__":
    main()