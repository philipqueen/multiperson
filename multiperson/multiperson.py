from typing import Tuple
import cv2
import numpy as np
from pathlib import Path

from data_models.camera_collection import CameraCollection, Camera
from utilities.get_synchronized_frames import display_frames, get_synchronized_frames


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
    constraint_values = np.array([x2.T @ fundamental @ x1 for x1, x2 in zip(points_0, points_1)])
    print("Epipolar constraint statistics, should be near 0:")
    print(f"Epipolar constraint mean: {np.mean(constraint_values)}")
    print(f"Epipolar constraint median: {np.median(constraint_values)}")
    print(f"Epipolar constraint std: {np.std(constraint_values)}")
    print(f"Epipolar constraint max: {np.max(constraint_values)} at {np.argmax(constraint_values)}")
    print(f"Epipolar constraint min: {np.min(constraint_values)} at {np.argmin(constraint_values)}")


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
    return (
        np.linalg.inv(camera_1_instrinsic).T
        @ essential
        @ np.linalg.inv(camera_0_instrinsic)
    )


def fundamental_from_camera_pair(camera_0: Camera, camera_1: Camera) -> np.ndarray:
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


def get_frames(cameras: list[Camera], active_frame: int) -> list[np.ndarray]:
    """
    This will obviously look very different in practice
    """
    if active_frame == 500: # use cached values
        frames = [get_saved_frame(camera, active_frame) for camera in cameras]
    else:
        video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/")
        synchronized_frames = get_synchronized_frames(video_path, active_frame)
        key_map = {Path(filename).stem: filename for filename in synchronized_frames.keys()}
        frames = [synchronized_frames[key_map[camera.name]] for camera in cameras]

    return frames

def get_saved_frame(camera: Camera, active_frame: int):
    """
    This will obviously look very different in practice
    """
    image_path = f"/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/{camera.name}_{active_frame}.jpg"
    image = cv2.imread(image_path)
    return image


def draw_and_display_lines(
    image_a: np.ndarray,
    image_b: np.ndarray,
    image_a_points: np.ndarray,
    image_b_points: np.ndarray,
    image_a_lines: np.ndarray,
    image_b_lines: np.ndarray,
):
    image_1_with_lines = draw_lines(image_a.copy(), image_a_lines, image_a_points)
    image_0_with_lines = draw_lines(image_b.copy(), image_b_lines, image_b_points)

    display_frames(
        frames={"image_0": image_0_with_lines, "image_1": image_1_with_lines}
    )


if __name__ == "__main__":
    path_to_calibration_toml = (
        Path(__file__).parent
        / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    )
    camera_collection = CameraCollection.from_file(path_to_calibration_toml)

    id_list = camera_collection.ids
    a_index = 0
    b_index = 2

    camera_a = camera_collection.by_id(id_list[a_index])
    camera_b = camera_collection.by_id(id_list[b_index])

    fundamental = fundamental_from_camera_pair(camera_a, camera_b)

    # Get image points:
    body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    body_data_cams_frame_points_xy = np.load(body_data_path)

    # Everything above this only has to happen once per pair of cameras
    # Everything below this will have to happen per frame

    active_frame = 200  # use 500 for cached data

    image_a, image_b = get_frames([camera_a, camera_b], active_frame)

    image_a_points = homogenize_points(
        body_data_cams_frame_points_xy[a_index, active_frame, :33, :2]
    )
    image_b_points = homogenize_points(
        body_data_cams_frame_points_xy[b_index, active_frame, :33, :2]
    )

    check_fundamental_epipolar_constraint(fundamental, image_a_points, image_b_points)

    image_b_lines = calculate_epipolar_lines(fundamental, image_a_points)
    image_a_lines = calculate_epipolar_lines(fundamental.T, image_b_points)

    distance_a = calculate_distance_to_lines(image_a_points, image_a_lines)
    distance_b = calculate_distance_to_lines(image_b_points, image_b_lines)

    print(f"distance_a average: {np.nanmean(distance_a)}")
    print(f"distance_b average: {np.nanmean(distance_b)}")

    draw_and_display_lines(
        image_a,
        image_b,
        image_a_points,
        image_b_points,
        image_a_lines,
        image_b_lines,
    )
