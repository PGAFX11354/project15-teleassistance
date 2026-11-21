"""Simulated daily routines built from the 6 activities of the smartphone
HAR dataset. A day = 96 slots of 15 minutes. Used to train and test the RNN
of Agent 2 (the dataset itself has no full days, only short activity blocks)."""
import numpy as np

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
A = {name: i for i, name in enumerate(ACTIVITIES)}
SLOTS_PER_DAY = 96
INACTIVE = {A["SITTING"], A["LAYING"]}

# Probabilities while awake, in the order: WALKING, UP, DOWN, SITTING, STANDING
_PERIODS = [
    (10, [0.30, 0.025, 0.025, 0.30, 0.35]),   # morning
    (17, [0.30, 0.025, 0.025, 0.40, 0.25]),   # daytime
    (24, [0.17, 0.015, 0.015, 0.60, 0.20]),   # evening
]


def normal_day(rng):
    """One normal day: sleeps at night, moves around and sits during the day."""
    wake = rng.uniform(6, 8)
    bed = rng.uniform(21.5, 23.5)
    day = np.empty(SLOTS_PER_DAY, dtype=int)
    prev = A["LAYING"]
    for s in range(SLOTS_PER_DAY):
        hour = s / 4
        if hour < wake or hour >= bed:                       # night
            cur = A["LAYING"]
            if rng.random() < 0.03:                          # short bathroom trip
                cur = rng.choice([A["STANDING"], A["WALKING"]])
        else:                                                # awake
            stay = 0.2 if prev in (A["WALKING_UPSTAIRS"], A["WALKING_DOWNSTAIRS"]) else 0.75
            if prev != A["LAYING"] and rng.random() < stay:
                cur = prev
            else:
                probs = next(p for limit, p in _PERIODS if hour < limit)
                cur = rng.choice(5, p=probs)
        day[s] = cur
        prev = cur
    return day


def inactive_afternoon(rng):
    """Anomaly: lying down from 10:00 to 16:00 (fall / unwell)."""
    day = normal_day(rng)
    day[40:64] = A["LAYING"]
    return day


def awake_at_night(rng):
    """Anomaly: walking around between 01:00 and 04:00."""
    day = normal_day(rng)
    day[4:16] = rng.choice([A["WALKING"], A["STANDING"]], size=12, p=[0.7, 0.3])
    return day


# Scenario names used by the Streamlit app (app.py)
SCENARIOS = {
    "Activité normale": normal_day,
    "Chute + Inactivité": inactive_afternoon,
}


WINDOWS_PER_SLOT = 15   # one sensor window per minute inside a 15-minute slot


def observe(day, pools, rng):
    """What the sensors + classifier 'see' for each slot of a TRUE day.
    For every slot we draw 15 real sensor windows of that activity from the
    dataset (one per minute), keep what the classifier predicted for each one,
    and take the majority vote. So real classifier mistakes (sitting vs
    standing...) can still end up in the sequence, but isolated errors are voted out.
    `pools[a]` = predicted activities (ints) for all real windows of activity a."""
    seen = np.empty_like(day)
    for a in np.unique(day):
        idx = np.where(day == a)[0]
        pool = pools[int(a)]
        draws = pool[rng.integers(0, len(pool), (len(idx), WINDOWS_PER_SLOT))]
        votes = np.stack([(draws == k).sum(axis=1) for k in range(len(ACTIVITIES))], axis=1)
        seen[idx] = votes.argmax(axis=1)
    return seen
