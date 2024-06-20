import numpy as np


def calculate_distance_to_lines(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidian distance from points to respective lines.
    Return nan value if denominator is 0.
    """
    if points.shape != lines.T.shape:
        raise ValueError("Points and lines must have transposed shapes (N, M) and (M, N)")

    # Calculates the absolute value of (a*x + b*y + c), then divides by the square root of the square
    numerators = np.abs(np.sum(points * lines.T, axis=1))
    denominators = np.sqrt(np.sum(lines[:2, :] ** 2, axis=0))  # as long as we normalize the lines, should always be 1s

    return numerators / denominators