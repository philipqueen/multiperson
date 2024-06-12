# Compute fundamental matrix:
"""
All of the example equations I can find assume no rotation or translation for cam 0
so likely we need to calculate the relative rotation and translation between the two cameras
and use that as camera 1's rotation and translation
# """
# rt_0 = np.hstack([np.eye(3), np.zeros((3, 1))])
# perspective_0 = np.dot(camera_0_instrinsic, rt_0)

# # rt_1 = np.hstack([camera_1_rotation, camera_1_translation.reshape(3, 1)])
# # perspective_1 = np.dot(camera_1_instrinsic, rt_1)
# rt_1 = np.hstack([relative_rotation, relative_translation.reshape(3, 1)])
# perspective_1 = np.dot(camera_1_instrinsic, rt_1)

# fundamental = fundamental_from_projections(perspective_0, perspective_1)

# print(f"fundamental: {fundamental}")
# check_fundamental_properties(fundamental)


import numpy as np


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