from src.test_runner.scenario_loader import ScenarioLoader


def main():
    loader = ScenarioLoader()

    print("Available scenarios:")
    for name in loader.list_scenarios():
        print("-", name)

    scenario = loader.load("warning_comm_delay.json")

    print("\nLoaded scenario:")
    print("name =", scenario.name)
    print("description =", scenario.description)
    print("initial_actions =", scenario.initial_actions)
    print("expected_final_state =", scenario.expected_final_state)
    print("expected_alert_level =", scenario.expected_alert_level)

    print("\nEvents:")
    for event in scenario.events:
        print(
            f"step={event.step}, desc={event.description}, inputs={event.inputs}"
        )


if __name__ == "__main__":
    main()