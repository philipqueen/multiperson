import numpy as np

from multiperson.geometry.epipolar_geometry import calculate_relative_rotation_and_translation


def test_identity_rotation_and_zero_translation():
    camera_0_rotation = np.eye(3)
    camera_0_translation = np.zeros(3)
    camera_1_rotation = np.eye(3)
    camera_1_translation = np.zeros(3)

    expected_rotation = np.eye(3)
    expected_translation = np.zeros(3)

    relative_rotation, relative_translation = calculate_relative_rotation_and_translation(
        camera_0_rotation, camera_0_translation, camera_1_rotation, camera_1_translation
    )

    assert np.allclose(relative_rotation, expected_rotation)
    assert np.allclose(relative_translation, expected_translation)

def test_translation():
    camera_0_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    camera_0_translation = np.array([1, 2, 3])
    camera_1_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    camera_1_translation = np.array([4, 5, 6])

    expected_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected_translation = np.array([3, 3, 3])

    relative_rotation, relative_translation = calculate_relative_rotation_and_translation(
        camera_0_rotation, camera_0_translation, camera_1_rotation, camera_1_translation
    )

    assert np.allclose(relative_rotation, expected_rotation)
    assert np.allclose(relative_translation, expected_translation)

def test_rotation_around_x():
    camera_0_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # 90 degree rotation around x
    camera_0_translation = np.array([1, 0, 0])
    camera_1_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) # 180 degree rotation around x
    camera_1_translation = np.array([0, 1, 0])

    expected_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # 90 degree rotation around x
    expected_translation = np.array([-1, 1, 0])

    relative_rotation, relative_translation = calculate_relative_rotation_and_translation(
        camera_0_rotation, camera_0_translation, camera_1_rotation, camera_1_translation
    )

    assert np.allclose(relative_rotation, expected_rotation)
    assert np.allclose(relative_translation, expected_translation)

def test_rotation_around_y():
    camera_0_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) # 90 degree rotation around y
    camera_0_translation = np.array([0, 1, 0])
    camera_1_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # -180 degree rotation around y
    camera_1_translation = np.array([0, 0, 1])

    expected_rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) # -90 degree rotation around y
    expected_translation = np.array([0, -1, 1])

    relative_rotation, relative_translation = calculate_relative_rotation_and_translation(
        camera_0_rotation, camera_0_translation, camera_1_rotation, camera_1_translation
    )

    assert np.allclose(relative_rotation, expected_rotation)
    assert np.allclose(relative_translation, expected_translation)

def test_rotation_around_z():
    camera_0_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # -90 degree rotation around z
    camera_0_translation = np.array([0, 0, 1])
    camera_1_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # 90 degree rotation around z
    camera_1_translation = np.array([0, 1, 0])

    expected_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) # 180 degree rotation around z
    expected_translation = np.array([0, 1, -1])

    relative_rotation, relative_translation = calculate_relative_rotation_and_translation(
        camera_0_rotation, camera_0_translation, camera_1_rotation, camera_1_translation
    )

    assert np.allclose(relative_rotation, expected_rotation)
    assert np.allclose(relative_translation, expected_translation)
