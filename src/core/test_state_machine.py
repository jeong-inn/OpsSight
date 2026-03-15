from src.core.state_machine import OperationalStateMachine, SystemEvent


def main():
    sm = OperationalStateMachine()

    steps = [
        (SystemEvent.BOOT_COMPLETE, "system boot completed"),
        (SystemEvent.START_COMMAND, "operator armed the system"),
        (SystemEvent.START_COMMAND, "operator activated the system"),
        (SystemEvent.WARNING_FAULT, "signal drift detected"),
        (SystemEvent.CRITICAL_FAULT, "compound fault escalated"),
        (SystemEvent.RESET_COMMAND, "manual reset after safe shutdown"),
    ]

    for event, reason in steps:
        result = sm.trigger(event, reason)
        print(
            f"[{event.value}] "
            f"{result.previous_state.value} -> {result.current_state.value} | "
            f"allowed={result.allowed} | changed={result.changed} | reason={result.reason}"
        )

    print("\nHistory:")
    for row in sm.get_history():
        print(row)


if __name__ == "__main__":
    main()