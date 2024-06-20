import numpy as np

from multiperson.geometry.epipolar_geometry import essential_from_rotation_and_translation, skew_symmetric_matrix_from_vector


def test_essential_from_rotation_and_translation_case_1():
    rotation = np.array(
        [
            [0.20675307718182576, 0.4531800366089544, -0.8671107308152476],
            [-0.40385972908878415, 0.8467634252944929, 0.3462499397889792],
            [0.891151212924148, 0.27860286431163017, 0.35809211063799695],
        ]
    )
    translation = np.array([1380.6399413475915, -778.4099209899186, 2034.333219500457])
    expected_essential = np.array(
        [
            [127.90431766141248, -1939.4661987309617, -983.13020631159],
            [-809.7543051985007, 537.2689606455491, -2258.388435311056],
            [-396.6462262082656, 1521.8352423248589, -196.92109890154416],
        ]
    )

    assert np.allclose(
        essential_from_rotation_and_translation(rotation, translation),
        expected_essential,
    )


def test_essential_from_rotation_and_translation_case_2():
    rotation = np.array(
        [
            [0.7928001064567758, 0.1584455723227262, -0.5885261182084011],
            [-0.21937262991812972, 0.9750829100444843, -0.032999511844623526],
            [0.56863313343924, 0.15526853883032415, 0.8078044567867694],
        ]
    )
    translation = np.array([1116.419984976144, -282.3084386926537, 808.5207324993025])
    expected_essential = np.array(
        [
            [16.8373873415614, -832.2083674519965, -201.36922547570566],
            [6.161928506783411, -45.23836959257649, -1377.6846076983747],
            [-21.097827948193565, 1133.3325699225506, -202.9872040790565],
        ]
    )

    assert np.allclose(
        essential_from_rotation_and_translation(rotation, translation),
        expected_essential,
    )


def test_skew_symmetric_matrix_from_vector_case_1():
    vector = np.array([1, 2, 3])
    expected_matrix = np.array(
        [
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0],
        ]
    )
    assert np.allclose(skew_symmetric_matrix_from_vector(vector), expected_matrix)


def test_skew_symmetric_matrix_from_vector_case_2():
    vector = np.array([-6.07, 2.92, 5.04])
    expected_matrix = np.array(
        [
            [0, -5.04, 2.92],
            [5.04, 0, 6.07],
            [-2.92, -6.07, 0],
        ]
    )
    assert np.allclose(skew_symmetric_matrix_from_vector(vector), expected_matrix)
