import numpy as np


def homogenize_single_frame_points(image_points: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates to homogeneous coordinates (x, y, 1).

    Takes a 2 dimensional array of shape (N, 2) or (N, 3) and returns a 3 dimensional array of shape (N, 3)
    """
    if image_points.ndim != 2 or image_points.shape[1] not in {2, 3}:
        raise ValueError("Input points must be 2D (N, 2) or 3D (N, 3)")
    
    image_points = image_points[:, :2]  # Take only x and y

    homogeneous_points = np.hstack(
        (image_points, np.ones((image_points.shape[0], 1)))
    )

    return homogeneous_points

def homogenize_data_array(image_points: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates to homogeneous coordinates (x, y, 1).

    Takes a 4 dimensional array of shape (A, B, C, 2) or (A, B, C, 3) and returns a 4 dimensional array of shape (A, B, C, 3)
    """
    if image_points.ndim != 4 or image_points.shape[3] not in {2, 3}:
        raise ValueError("Input points must be shape (A, B, C, 2) or (A, B, C, 3)")
    
    image_points = image_points[:, :, :, :2]  # Take only x and y

    homogeneous_points = np.concatenate(
        (image_points, np.ones((image_points.shape[0], image_points.shape[1], image_points.shape[2], 1))), axis=3
    )

    return homogeneous_points