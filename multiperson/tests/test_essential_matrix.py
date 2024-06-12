import numpy as np
from multiperson.multiperson import skew_symmetric_matrix_from_vector


def test_skew_symmetric_matrix_from_vector():
    vector = np.array([1, 2, 3])
    expected_matrix = np.array(
        [
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0],
        ]
    )
    assert np.allclose(skew_symmetric_matrix_from_vector(vector), expected_matrix)

    vector = np.array([-6.07, 2.92, 5.04])
    expected_matrix = np.array(
        [
            [0, -5.04, 2.92],
            [5.04, 0, 6.07],
            [-2.92, -6.07, 0],
        ]
    )
    assert np.allclose(skew_symmetric_matrix_from_vector(vector), expected_matrix)