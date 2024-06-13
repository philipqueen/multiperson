import re
import numpy as np
import pytest
from multiperson.multiperson import calculate_distance_to_lines, homogenize_points

def test_basic_functionality():
    points = homogenize_points(np.array([[1, 2], [2, 4], [3, 6]]))
    lines = np.array([[2, 1, -2], [2, 1, -4], [2, 1, -4]]).T
    expected_distances = np.array([0.89442719, 1.78885438, 3.57770876])
    distances = calculate_distance_to_lines(points, lines)
    assert np.allclose(distances, expected_distances, rtol=1e-6)

def test_edge_case_on_the_line():
    points = homogenize_points(np.array([[1, 2], [2, 4], [3, 8]]))
    lines = np.array([[2, 1, -4], [2, 1, -8], [3, -1, -1]]).T
    expected_distances = np.array([0., 0., 0.])
    distances = calculate_distance_to_lines(points, lines)
    assert np.allclose(distances, expected_distances, rtol=1e-6)

def test_invalid_input():
    points = homogenize_points(np.array([[1, 1], [2, 2]]))
    lines = np.array([[1, -1, 0]]).T
    with pytest.raises(ValueError, match=re.escape("Points and lines must have transposed shapes (N, M) and (M, N)")):
        calculate_distance_to_lines(points, lines)

@pytest.mark.filterwarnings("ignore:invalid value")  # suppresses division by zero warning
def test_zero_denominator():
    points = homogenize_points(np.array([[2, 2], [1, 1], [3, 3]]))
    lines = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T
    distances = calculate_distance_to_lines(points, lines)
    assert np.isnan(distances).all()