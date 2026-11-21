"""Print simulated days as readable timelines.  Usage: python -m training.show_days"""
import numpy as np

from agents.routine_sim import (ACTIVITIES, awake_at_night, inactive_afternoon,
                                normal_day)


def timeline(day):
    """Group consecutive identical slots: '00:00-06:45  LAYING'."""
    blocks, start = [], 0
    for i in range(1, len(day) + 1):
        if i == len(day) or day[i] != day[start]:
            h0, m0 = divmod(start * 15, 60)
            h1, m1 = divmod(i * 15 - 15, 60)
            blocks.append(f"  {h0:02d}:{m0:02d}-{h1:02d}:{m1:02d}  {ACTIVITIES[day[start]]}")
            start = i
    return "\n".join(blocks)


rng = np.random.default_rng(1)
for title, make in [("NORMAL DAY", normal_day),
                    ("NORMAL DAY (another one)", normal_day),
                    ("ANOMALY: inactive afternoon", inactive_afternoon)]:
    day = make(rng)
    print(f"\n=== {title} ===")
    print(timeline(day))
    print("  raw slots:", day[:24].tolist(), "...")
