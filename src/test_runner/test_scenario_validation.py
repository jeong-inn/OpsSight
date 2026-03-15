from src.test_runner.scenario_loader import ScenarioLoader
from src.test_runner.scenario_executor import ScenarioExecutor
from src.test_runner.validator import ScenarioValidator


def main():
    loader = ScenarioLoader()
    executor = ScenarioExecutor()
    validator = ScenarioValidator()

    for filename in loader.list_scenarios():
        scenario = loader.load(filename)
        execution = executor.run(scenario)
        validation = validator.validate(
            expected_final_state=scenario.expected_final_state,
            actual_final_state=execution.final_state,
            expected_alert_level=scenario.expected_alert_level,
            actual_alert_level=execution.final_alert_level,
        )

        print(f"\n=== {scenario.name} ===")
        print("expected_final_state =", scenario.expected_final_state)
        print("actual_final_state   =", execution.final_state)
        print("expected_alert_level =", scenario.expected_alert_level)
        print("actual_alert_level   =", execution.final_alert_level)
        print("passed =", validation.passed)
        print("details =", validation.details)

        print("step_results:")
        for step in execution.step_results:
            print(step)


if __name__ == "__main__":
    main()