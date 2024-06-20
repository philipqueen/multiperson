import numpy as np


def homogenize_points(image_points: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates to homogeneous coordinates (x, y, 1).
    """
    if image_points.ndim != 2 or image_points.shape[1] not in {2, 3}:
        raise ValueError("Input points must be 2D (N, 2) or 3D (N, 3)")
    
    image_points = image_points[:, :2]  # Take only x and y

    homogeneous_points = np.hstack(
        (image_points, np.ones((image_points.shape[0], 1)))
    )

    return homogeneous_points