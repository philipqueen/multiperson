from typing import Tuple
import cv2
import numpy as np


from multiperson.data_models.camera_collection import Camera


def check_fundamental_properties(fundamental: np.ndarray) -> None:
    """
    Check fundamental matrix has required properties:
    - rank 2
    - det(F) = 0

    Raises ValueError if any of these properties are violated
    """
    if fundamental.shape != (3, 3):
        raise ValueError("Fundamental matrix must be 3x3")

    # Compute the SVD of F
    _, S, _ = np.linalg.svd(fundamental)
    print("Singular values of F:", S)

    # "Geometrically, F represents a mapping from the 2-dimensional projective plane IP2 of the first image to the pencil of epipolar lines through the epipole e′. Thus, it represents a mapping from a 2-dimensional onto a 1-dimensional projective space, and hence must have rank 2." - HZ book
    rank = np.sum(S > 1e-10)  # Count singular values significantly greater than zero
    print("Rank of F:", rank)
    if rank != 2:
        raise ValueError(f"Fundamental matrix is rank {rank}, but must be 2")

    # check that det(F) = 0: "F also satisfies the constraint det F = 0" - HZ book
    # technically, this is redundant because for a 3x3 matrix, if rank < 3, then det(F) = 0, but could be useful for slight numerical differences
    determinant = np.linalg.det(fundamental)
    print("Determinant of F:", determinant)
    if abs(determinant) > 1e-10:
        raise ValueError("Fundamental matrix must have a determinant of 0")


def check_fundamental_epipolar_constraint(
    fundamental: np.ndarray, points_0: np.ndarray, points_1: np.ndarray
):
    # Check the epipolar constraint for each pair of points
    # "The fundamental matrix satisfies the condition that for any pair of corresponding points x ↔ x′ in the two images x′TFx = 0." - HZ book
    constraint_values = np.array(
        [x2.T @ fundamental @ x1 for x1, x2 in zip(points_0, points_1)]
    )
    print("Epipolar constraint statistics, should be near 0:")
    print(f"Epipolar constraint mean: {np.mean(constraint_values)}")
    print(f"Epipolar constraint median: {np.median(constraint_values)}")
    print(f"Epipolar constraint std: {np.std(constraint_values)}")
    print(
        f"Epipolar constraint max: {np.max(constraint_values)} at {np.argmax(constraint_values)}"
    )
    print(
        f"Epipolar constraint min: {np.min(constraint_values)} at {np.argmin(constraint_values)}"
    )


def calculate_epipolar_lines(
    fundamental_matrix: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Calculate the epipolar lines for a set of points using the fundamental matrix.
    Lines are represent in standard form, ax + by + c = 0, where a, b, and c are the first dimension of the output array.
    The lines represent matching points for the input points in the other image view, so points from image 0 give line sin image 1 and vice versa.
    """

    lines = fundamental_matrix @ points.T
    normalized_lines = lines / np.linalg.norm(lines[:2], axis=0)

    return normalized_lines


def essential_from_rotation_and_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return skew_symmetric_matrix_from_vector(translation) @ rotation


def skew_symmetric_matrix_from_vector(translation: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )


def fundamental_from_essential(
    camera_0_instrinsic: np.ndarray,
    camera_1_instrinsic: np.ndarray,
    essential: np.ndarray,
):
    """
    'The relationship between the fundamental and essential matrices is E = K`^TFK' - HZ book
    """
    return (
        np.linalg.inv(camera_1_instrinsic).T
        @ essential
        @ np.linalg.inv(camera_0_instrinsic)
    )


def calculate_relative_rotation_and_translation(
    camera_0_rotation: np.ndarray,
    camera_0_translation: np.ndarray,
    camera_1_rotation: np.ndarray,
    camera_1_translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # for rotation matrices, the inverse is the transpose. These methods should be the same, but differ because of slight numerical errors
    # relative_rotation = np.linalg.inv(camera_0_rotation) @ camera_1_rotation
    relative_rotation = (
        camera_1_rotation @ camera_0_rotation.T
    )  # this version appears more accurate across camera views
    relative_translation = camera_1_translation - (
        relative_rotation @ camera_0_translation
    )

    return relative_rotation, relative_translation


def fundamental_from_camera_pair(camera_0: Camera, camera_1: Camera) -> np.ndarray:
    # TODO: See if Singular Value Thresholding as a denoise method could help here
    relative_rotation, relative_translation = (
        calculate_relative_rotation_and_translation(
            camera_0.rotation,
            camera_0.translation,
            camera_1.rotation,
            camera_1.translation,
        )
    )

    essential = essential_from_rotation_and_translation(
        relative_rotation, relative_translation
    )

    fundamental = fundamental_from_essential(
        camera_0.intrinsic, camera_1.intrinsic, essential
    )

    print(f"fundamental: {fundamental}")
    check_fundamental_properties(fundamental)

    return fundamental
