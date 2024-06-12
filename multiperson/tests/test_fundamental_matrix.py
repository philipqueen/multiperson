import numpy as np
import pytest
from multiperson.multiperson import fundamental_from_essential, check_fundamental_properties


def test_check_fundamental_properties():
    # make sure it throws error if isn't rank 2, or isn't det 0
    rank_1 = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    rank_2 = np.array([[1, 2, 3], [2, 4, 6], [9, 4, 14]])
    rank_3 = np.array([[1, 2, 3], [9, 4, 14], [12, 30, -4]])
    undersized = np.random.random((2, 2))
    oversized = np.random.random((4, 4))
    
    with pytest.raises(ValueError, match=f"Fundamental matrix is rank {1}, but must be 2"):
        check_fundamental_properties(rank_1)

    with pytest.raises(ValueError, match=f"Fundamental matrix is rank {3}, but must be 2"):
        check_fundamental_properties(rank_3)

    with pytest.raises(ValueError, match="Fundamental matrix must be 3x3"):
        check_fundamental_properties(oversized)

    with pytest.raises(ValueError, match="Fundamental matrix must be 3x3"):
        check_fundamental_properties(undersized)

    try:
        check_fundamental_properties(rank_2)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

    # it's impossible to do a negative check of the determinant, since any det != 0 is caught by the rank check

def test_fundamental_from_essential():
    essential = np.array(
        [
            [127.90431766141248, -1939.4661987309617, -983.13020631159],
            [-809.7543051985007, 537.2689606455491, -2258.388435311056],
            [-396.6462262082656, 1521.8352423248589, -196.92109890154416],
        ]
    )
    intrinsic_1 = np.array(
        [[883.5893169135845, 0.0, 359.5], [0.0, 883.5893169135845, 639.5], [0.0, 0.0, 1.0]]
    )
    intrinsic_2 = np.array(
        [[883.5893169135845, 0.0, 359.5], [0.0, 883.5893169135845, 639.5], [0.0, 0.0, 1.0]]
    )

    fundamental = fundamental_from_essential(intrinsic_1, intrinsic_2, essential)

    try:
        check_fundamental_properties(fundamental)
    except Exception as e:
        pytest.fail(f"Error in check_fundamental_properties: {e}")

    # check that we can convert back to essential
    back_to_essential = intrinsic_1.T @ fundamental @ intrinsic_2
    assert np.allclose(essential, back_to_essential)

    # test epipolar constraint somehow?
