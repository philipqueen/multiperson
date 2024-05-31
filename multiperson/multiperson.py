import cv2
import numpy as np
from pathlib import Path

from utilities.read_calibration_toml import read_calibration_toml
from utilities.get_synchronized_frames import display_frames


def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Calculate the fundamental matrix from the projection matrices (P1, P2).

    Adapted and modified from OpenCV's fundamentalFromProjections function in the opencv_sfm module
    """

    assert P1.shape == (3, 4), "P1 must be a 3x4 matrix"
    assert P2.shape == (3, 4), "P2 must be a 3x4 matrix"

    fundamental = np.zeros((3, 3))

    # for i in range(3):
    #     for j in range(3):
    #         x_matrix = np.delete(P1, i, axis=0)
    #         y_matrix = np.delete(P2, j, axis=0)
    #         xy_matrix = np.vstack((x_matrix, y_matrix))
    #         fundamental[i, j] = np.linalg.det(xy_matrix)

    # return fundamental

    x_matrix = np.array(
        [
            np.vstack((P1[1, :], P1[2, :])),
            np.vstack((P1[2, :], P1[0, :])),
            np.vstack((P1[0, :], P1[1, :])),
        ]
    )

    y_matrix = np.array(
        [
            np.vstack((P2[1, :], P2[2, :])),
            np.vstack((P2[2, :], P2[0, :])),
            np.vstack((P2[0, :], P2[1, :])),
        ]
    )

    for i in range(3):
        for j in range(3):
            xy_matrix = np.vstack((x_matrix[j], y_matrix[i]))
            fundamental[i, j] = np.linalg.det(xy_matrix)

    return fundamental


def check_fundamental(
    fundamental: np.ndarray, points_0: np.ndarray, points_1: np.ndarray
):
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
        raise ValueError("Fundamental matrix is not invertible")

    # Check the epipolar constraint for each pair of points
    # "The fundamental matrix satisfies the condition that for any pair of corresponding points x ↔ x′ in the two images x′TFx = 0." - HZ book
    i = 0
    for x1, x2 in zip(points_0, points_1):
        epipolar_constraint = x2.T @ fundamental @ x1
        print(
            f"Epipolar constraint for point pair {i}: {epipolar_constraint} - should be 0"
        )
        i += 1


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

    lines = fundamental_matrix @ points.T
    normalized_lines = lines / np.linalg.norm(lines[:2], axis=0)

    return normalized_lines


def essential_from_rotation_and_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return get_skew_symmetric_translation(translation) @ rotation


def get_skew_symmetric_translation(translation: np.ndarray) -> np.ndarray:
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
    'The relationship between the fundamental and essential matrices is E = K`TFK' - HZ book
    """
    return (
        np.linalg.inv(camera_1_instrinsic).T
        @ essential
        @ np.linalg.inv(camera_0_instrinsic)
    )


def homogenize_points(image_points: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates to homogeneous coordinates (x, y, 1).
    """
    if image_points.shape[1] == 3:
        image_points = image_points[:, :2]  # Take only x and y
  
    if image_points.shape[1] == 2:
        homogeneous_points = np.hstack((image_points, np.ones((image_points.shape[0], 1))))
    else:
        raise ValueError("Input points must be 2D (N, 2) or 3D (N, 3)")

    return homogeneous_points

def calculate_distance_to_lines(
    points: np.ndarray, lines: np.ndarray
) -> np.ndarray:
    # Calculates the absolute value of (a*x + b*y + c)
    numerators = np.abs(np.sum(points * lines.T, axis=1))

    # (a^2 + b^2)^0.5
    denominators = np.sqrt(np.sum(lines[:2, :]**2, axis=0))

    return numerators / denominators


if __name__ == "__main__":
    # Setup:
    path_to_calibration_toml = (
        Path(__file__).parent
        / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    )
    calibration = read_calibration_toml(path_to_calibration_toml)

    image_0_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam2_500.jpg"
    image_0 = cv2.imread(image_0_path)

    image_1_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam3_500.jpg"
    image_1 = cv2.imread(image_1_path)

    camera_0_instrinsic = np.array(
        calibration["cam_0"]["instrinsics_matrix"]
    )  # TODO: just do these numpy ops when making the dict
    camera_1_instrinsic = np.array(calibration["cam_1"]["instrinsics_matrix"])
    camera_0_rotation, _ = cv2.Rodrigues(np.array(calibration["cam_0"]["rotation"]))
    camera_0_translation = np.array(calibration["cam_0"]["translation"])
    camera_1_rotation, _ = cv2.Rodrigues(np.array(calibration["cam_1"]["rotation"]))
    camera_1_translation = np.array(calibration["cam_1"]["translation"])

    # Compute relative rotation and translation
    relative_rotation = np.dot(camera_1_rotation, camera_0_rotation.T)
    relative_translation = camera_1_translation - np.dot(
        relative_rotation, camera_0_translation
    )

    # Compute fundamental matrix:
    """
    All of the example equations I can find assume no rotation or translation for cam 0
    so likely we need to calculate the relative rotation and translation between the two cameras
    and use that as camera 1's rotation and translation
    # """
    # rt_0 = np.c_[np.eye(3), np.array([0.0, 0.0, 0.0])]
    # perspective_0 = np.dot(camera_0_instrinsic, rt_0)

    # rt_1 = np.c_[camera_1_rotation, camera_1_translation]
    # perspective_1 = np.dot(camera_1_instrinsic, rt_1)

    # fundamental = fundamental_from_projections(perspective_0, perspective_1)

    # Compute skew-symmetric matrix of T
    essential = essential_from_rotation_and_translation(
        relative_rotation, relative_translation
    )

    # Compute fundamental matrix
    fundamental = fundamental_from_essential(
        camera_0_instrinsic, camera_1_instrinsic, essential
    )

    print(f"fundamental: {fundamental}")

    # Get image points:
    body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    body_data_cams_frame_points_xy = np.load(body_data_path)

    print(
        f"body_data_cams_frame_points_xy.shape: {body_data_cams_frame_points_xy.shape}"
    )

    active_frame = 500

    image_0_points = body_data_cams_frame_points_xy[0, active_frame, :33, :2]
    image_1_points = body_data_cams_frame_points_xy[1, active_frame, :33, :2]

    # Make sure points are homogenous
    image_0_points = homogenize_points(image_0_points)
    image_1_points = homogenize_points(image_1_points)

    print(f"image_0_points.shape: {image_0_points.shape}")
    print(f"image_1_points.shape: {image_1_points.shape}")

    check_fundamental(fundamental, image_0_points, image_1_points)

    image_1_lines = calculate_epipolar_lines(fundamental, image_0_points)
    image_0_lines = calculate_epipolar_lines(fundamental.T, image_1_points)

    print(f"image_1_lines.shape: {image_1_lines.shape}")
    print(f"image_0_lines.shape: {image_0_lines.shape}")

    image_1_with_lines = draw_lines(image_0.copy(), image_0_lines, image_0_points)
    image_0_with_lines = draw_lines(image_1.copy(), image_1_lines, image_1_points)

    display_frames(
        frames={"image_0": image_0_with_lines, "image_1": image_1_with_lines}
    )

    distance_0 = calculate_distance_to_lines(image_0_points, image_0_lines)
    distance_1 = calculate_distance_to_lines(image_1_points, image_1_lines)

    print(f"distance_0 average: {np.mean(distance_0)}")
    print(f"distance_1 average: {np.mean(distance_1)}")
