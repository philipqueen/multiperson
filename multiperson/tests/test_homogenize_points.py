import re
import numpy as np
import pytest

from multiperson.geometry.homogenize_points import homogenize_single_frame_points, homogenize_data_array


def testhomogenize_2d_points():
    points_2d = np.array([[1, 2], [3, 4], [5, 6]])
    expected_2d = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])
    result_2d = homogenize_single_frame_points(points_2d)
    np.testing.assert_array_equal(result_2d, expected_2d)


def test_homogenize_3d_points():
    points_3d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_3d = np.array([[1, 2, 1], [4, 5, 1], [7, 8, 1]])
    result_3d = homogenize_single_frame_points(points_3d)
    np.testing.assert_array_equal(result_3d, expected_3d)


def test_homogenize_floating_points():
    points_mixed = np.array([[1.0, 2.0], [3.5, 4.5], [5.25, 6.75]])
    expected_mixed = np.array([[1.0, 2.0, 1.0], [3.5, 4.5, 1.0], [5.25, 6.75, 1.0]])
    result_mixed = homogenize_single_frame_points(points_mixed)
    np.testing.assert_array_equal(result_mixed, expected_mixed)


def test_homogenize_points_invalid_shape():
    with pytest.raises(
        ValueError, match=re.escape("Input points must be 2D (N, 2) or 3D (N, 3)")
    ):
        homogenize_single_frame_points(np.array([1, 2]))

    with pytest.raises(
        ValueError, match=re.escape("Input points must be 2D (N, 2) or 3D (N, 3)")
    ):
        homogenize_single_frame_points(np.array([[1, 2, 3, 4]]))

def testhomogenize_2d_array():
    points_2d = np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]])
    expected_2d = np.array([[[[1, 2, 1], [3, 4, 1], [5, 6, 1]], [[7, 8, 1], [9, 10, 1], [11, 12, 1]]]])
    result_2d = homogenize_data_array(points_2d)
    np.testing.assert_array_equal(result_2d, expected_2d)


def test_homogenize_3d_array():
    points_3d = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]])
    expected_3d = np.array([[[[1, 2, 1], [4, 5, 1], [7, 8, 1]], [[10, 11, 1], [13, 14, 1], [16, 17, 1]]]])
    result_3d = homogenize_data_array(points_3d)
    np.testing.assert_array_equal(result_3d, expected_3d)


def test_homogenize_floating_point_array():
    points_mixed = np.array([[[[1.0, 2.0], [3.5, 4.5], [5.25, 6.75]], [[7.0, 8.0], [9.5, 10.5], [11.25, 12.75]]]])
    expected_mixed = np.array([[[[1.0, 2.0, 1.0], [3.5, 4.5, 1.0], [5.25, 6.75, 1.0]], [[7.0, 8.0, 1.0], [9.5, 10.5, 1.0], [11.25, 12.75, 1.0]]]])
    result_mixed = homogenize_data_array(points_mixed)
    np.testing.assert_array_equal(result_mixed, expected_mixed)


def test_homogenize_array_invalid_shape():
    with pytest.raises(
        ValueError, match=re.escape("Input points must be shape (A, B, C, 2) or (A, B, C, 3)")
    ):
        homogenize_data_array(np.array([[1, 2]]))

    with pytest.raises(
        ValueError, match=re.escape("Input points must be shape (A, B, C, 2) or (A, B, C, 3)")
    ):
        homogenize_data_array(np.array([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]))
