import sys
import time
from datetime import datetime

from .event_generator import (
    generate_normal_event,
    generate_suspicious_event,
)


def print_event(event):
    print(
        f"[{event.timestamp.strftime('%H:%M:%S')}] "
        f"{event.user_id:<10} → "
        f"{event.event_type.value:<18} "
        f"{event.action or ''}"
    )


def run_simulation(scenario: str):
    user_id = "EMP_001"

    print("=" * 60)
    print(f"INSIDER THREAT ACTIVITY SIMULATOR")
    print(f"Scenario: {scenario.upper()}")
    print("=" * 60)

    current_time = datetime.now()

    generator = (
        generate_suspicious_event
        if scenario == "suspicious"
        else generate_normal_event
    )

    for _ in range(15):
        event = generator(user_id, current_time)

        print_event(event)

        current_time = current_time.replace(
            second=(current_time.second + 2) % 60
        )

        time.sleep(1)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if scenario not in {"normal", "suspicious"}:
        print("Usage: python simulator.py [normal|suspicious]")
        sys.exit(1)

    run_simulation(scenario)