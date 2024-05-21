import cv2
import numpy as np
from pathlib import Path

from utilities.read_calibration_toml import read_calibration_toml


def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Calculate the fundamental matrix from the projection matrices (P1, P2).

    Adapted and modified from OpenCV's fundamentalFromProjections function in the opencv_sfm module
    """

    assert P1.shape == (3, 4), "P1 must be a 3x4 matrix"
    assert P2.shape == (3, 4), "P2 must be a 3x4 matrix"

    F = np.zeros((3, 3))

    X = np.array(
        [
            np.vstack((P1[1, :], P1[2, :])),
            np.vstack((P1[2, :], P1[0, :])),
            np.vstack((P1[0, :], P1[1, :])),
        ]
    )

    Y = np.array(
        [
            np.vstack((P2[1, :], P2[2, :])),
            np.vstack((P2[2, :], P2[0, :])),
            np.vstack((P2[0, :], P2[1, :])),
        ]
    )

    for i in range(3):
        for j in range(3):
            XY = np.vstack((X[j], Y[i]))
            F[i, j] = np.linalg.det(XY)

    return F


def draw_epipolar_lines(
    image_1: np.ndarray,
    image_2: np.ndarray,
    fundamental_matrix: np.ndarray,
    points_1: np.ndarray,
    points_2: np.ndarray,
):
    points_1 = np.hstack([points_1, np.ones((points_1.shape[0], 1))])
    points_2 = np.hstack([points_2, np.ones((points_2.shape[0], 1))])

    lines2 = fundamental_matrix @ points_1.T
    lines2 /= np.linalg.norm(lines2[:2], axis=0)

    lines1 = fundamental_matrix.T @ points_2.T
    lines1 /= np.linalg.norm(lines1[:2], axis=0)

    def draw_lines(img, lines, pts):
        r, c = img.shape[:2]
        for r, pt in zip(lines.T, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(pt[:2]), 5, color, -1)
        return img

    img1_with_lines = draw_lines(image_1.copy(), lines1, points_1)
    img2_with_lines = draw_lines(image_2.copy(), lines2, points_2)

    return img1_with_lines, img2_with_lines


if __name__ == "__main__":
    path_to_calibration_toml = (
        Path(__file__).parent
        / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    )
    calibration = read_calibration_toml(path_to_calibration_toml)

    image_0_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam2_500.jpg"
    image_0 = cv2.imread(image_0_path)

    image_1_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/synchronized_frames/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam3_500.jpg"
    image_1 = cv2.imread(image_1_path)

    camera_0_instrinsic = np.array(calibration["cam_0"]["instrinsics_matrix"])
    camera_1_instrinsic = np.array(calibration["cam_1"]["instrinsics_matrix"])
    camera_0_rotation, _ = cv2.Rodrigues(np.array(calibration["cam_0"]["rotation"]))
    camera_0_translation = np.array(calibration["cam_0"]["translation"])
    camera_1_rotation, _ = cv2.Rodrigues(np.array(calibration["cam_1"]["rotation"]))
    camera_1_translation = np.array(calibration["cam_1"]["translation"])

    rt_0 = np.c_[camera_0_rotation, camera_0_translation]
    perspective_0 = np.dot(camera_0_instrinsic, rt_0)

    rt_1 = np.c_[camera_1_rotation, camera_1_translation]
    perspective_1 = np.dot(camera_1_instrinsic, rt_1)

    fundamental = fundamental_from_projections(perspective_0, perspective_1)

    print(fundamental)
