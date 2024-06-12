from typing import Tuple
import cv2
import numpy as np
from pathlib import Path

from data_models.camera_collection import CameraCollection, Camera
from utilities.get_synchronized_frames import display_frames


def check_fundamental_properties(fundamental: np.ndarray):
    # Compute the SVD of F
    _, S, _ = np.linalg.svd(fundamental)
    print("Singular values of F:", S)

    # "Geometrically, F represents a mapping from the 2-dimensional projective plane IP2 of the first image to the pencil of epipolar lines through the epipole e′. Thus, it represents a mapping from a 2-dimensional onto a 1-dimensional projective space, and hence must have rank 2." - HZ book
    rank = np.sum(S > 1e-10)  # Count singular values significantly greater than zero
    print("Rank of F:", rank)
    if rank != 2:
        raise ValueError(f"Fundamental matrix is rank {rank}, but must be 2")

    # check that det(F) = 0: "F also satisfies the constraint det F = 0" - HZ book
    determinant = np.linalg.det(fundamental)
    print("Determinant of F:", determinant)
    if abs(determinant) > 1e-10:
        raise ValueError("Fundamental matrix must have a determinant of 0")


def check_fundamental_epipolar_constraint(
    fundamental: np.ndarray, points_0: np.ndarray, points_1: np.ndarray
):
    # Check the epipolar constraint for each pair of points
    # "The fundamental matrix satisfies the condition that for any pair of corresponding points x ↔ x′ in the two images x′TFx = 0." - HZ book
    for x1, x2, i in zip(points_0, points_1, range(len(points_0))):
        epipolar_constraint = x2.T @ fundamental @ x1
        print(
            f"Epipolar constraint for point pair {i}: {epipolar_constraint} - should be 0"
        )  # check the units here - is this pixels or mm?


def draw_lines(image: np.ndarray, lines: np.ndarray, points: np.ndarray):
    rows, columns = image.shape[:2]
    for rows, point in zip(lines.T, points):
        # print(f"rows: {rows}, point: {point}")
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -rows[2] / rows[1]])
        x1, y1 = map(int, [columns, -(rows[2] + rows[0] * columns) / rows[1]])
        image = cv2.line(image, (x0, y0), (x1, y1), color, 1)
        image = cv2.circle(image, tuple(point[:2].astype(int)), 5, color, -1)
    return image


def calculate_epipolar_lines(
    fundamental_matrix: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Calculate the epipolar lines for a set of points using the fundamental matrix.
    Lines are represent in standard form, ax + by + c = 0, where a, b, and c are the first dimension of the output array.
    The lines represent matching points for the input points in the other image view, so points from image 0 give line sin image 1 and vice versa.
    """
    # TODO: Test (potentially compare to opencv's function cv2.computeCorrespondEpilines)

    lines = fundamental_matrix @ points.T
    normalized_lines = lines / np.linalg.norm(lines[:2], axis=0)

    return normalized_lines


def essential_from_rotation_and_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    # TODO: Test
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
    # TODO: Test
    return (
        np.linalg.inv(camera_1_instrinsic).T
        @ essential
        @ np.linalg.inv(camera_0_instrinsic)
    )


def fundamental_from_camera_pair(cam_0: Camera, cam_1: Camera) -> np.ndarray:
    relative_rotation, relative_translation = (
        calculate_relative_rotation_and_translation(
            cam_0.rotation,
            cam_0.translation,
            cam_1.rotation,
            cam_1.translation,
        )
    )

    essential = essential_from_rotation_and_translation(
        relative_rotation, relative_translation
    )

    fundamental = fundamental_from_essential(
        cam_0.intrinsic, cam_1.intrinsic, essential
    )

    print(f"fundamental: {fundamental}")
    check_fundamental_properties(fundamental)

    return fundamental


def homogenize_points(image_points: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates to homogeneous coordinates (x, y, 1).
    """
    # TODO: Test
    if image_points.shape[1] == 3:
        image_points = image_points[:, :2]  # Take only x and y

    if image_points.shape[1] == 2:
        homogeneous_points = np.hstack(
            (image_points, np.ones((image_points.shape[0], 1)))
        )
    else:
        raise ValueError("Input points must be 2D (N, 2) or 3D (N, 3)")

    return homogeneous_points


def calculate_distance_to_lines(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    # Calculates the absolute value of (a*x + b*y + c), then divides by the square root of the square
    # TODO: Test
    numerators = np.abs(np.sum(points * lines.T, axis=1))
    denominators = np.sqrt(np.sum(lines[:2, :] ** 2, axis=0))

    return numerators / denominators


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


def get_frame(camera: Camera, active_frame: int):
    """
    This will obviously look very different in practice
    """
    image_path = f"/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/{camera.name}_{active_frame}.jpg"
    image = cv2.imread(image_path)
    return image


if __name__ == "__main__":
    path_to_calibration_toml = (
        Path(__file__).parent
        / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    )
    camera_collection = CameraCollection.from_file(path_to_calibration_toml)

    id_list = camera_collection.ids
    a_index = 0
    b_index = 2

    cam_a = camera_collection.by_id(id_list[a_index])
    cam_b = camera_collection.by_id(id_list[b_index])

    fundamental = fundamental_from_camera_pair(cam_a, cam_b)

    # Get image points:
    body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    body_data_cams_frame_points_xy = np.load(body_data_path)

    print(
        f"body_data_cams_frame_points_xy.shape: {body_data_cams_frame_points_xy.shape}"
    )

    # Everything above this only has to happen once per pair of cameras
    # Everything below this will have to happen per frame

    active_frame = 500

    image_a = get_frame(cam_a, active_frame)
    image_b = get_frame(cam_b, active_frame)

    image_0_points = body_data_cams_frame_points_xy[a_index, active_frame, :33, :2]
    image_1_points = body_data_cams_frame_points_xy[b_index, active_frame, :33, :2]

    # Make sure points are homogenous
    image_0_points = homogenize_points(image_0_points)
    image_1_points = homogenize_points(image_1_points)

    print(f"image_0_points.shape: {image_0_points.shape}")
    print(f"image_1_points.shape: {image_1_points.shape}")

    check_fundamental_epipolar_constraint(fundamental, image_0_points, image_1_points)

    image_1_lines = calculate_epipolar_lines(fundamental, image_0_points)
    image_0_lines = calculate_epipolar_lines(fundamental.T, image_1_points)

    distance_0 = calculate_distance_to_lines(image_0_points, image_0_lines)
    distance_1 = calculate_distance_to_lines(image_1_points, image_1_lines)

    print(f"distance_0 average: {np.mean(distance_0)}")
    print(f"distance_1 average: {np.mean(distance_1)}")

    print(f"image_1_lines.shape: {image_1_lines.shape}")
    print(f"image_0_lines.shape: {image_0_lines.shape}")

    image_1_with_lines = draw_lines(image_a.copy(), image_0_lines, image_0_points)
    image_0_with_lines = draw_lines(image_b.copy(), image_1_lines, image_1_points)

    display_frames(
        frames={"image_0": image_0_with_lines, "image_1": image_1_with_lines}
    )
