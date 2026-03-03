DECELERATION = 8  # m/s²
REACTION_TIME = 0.9  # s


def stopping_distance(speed_kmh: float) -> float:
    """Total stopping distance in meters (reaction + braking) for a given speed in km/h."""
    v = speed_kmh / 3.6
    reaction_distance = v * REACTION_TIME
    braking_distance = v ** 2 / (2 * DECELERATION)
    return reaction_distance + braking_distance


def safety_index(following_distance: float, speed_kmh: float) -> float:
    """Ratio of following distance to stopping distance, capped at 1."""
    sd = stopping_distance(speed_kmh)
    if sd <= 0:
        return 1.0
    return min(following_distance / sd, 1.0)
