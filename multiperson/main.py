import cv2
import numpy as np
from pathlib import Path

from multiperson.data_models.camera_collection import CameraCollection, Camera
from multiperson.geometry.calculate_distance_to_lines import calculate_distance_to_lines
from multiperson.geometry.epipolar_geometry import calculate_epipolar_lines, check_fundamental_epipolar_constraint, fundamental_from_camera_pair
from multiperson.geometry.homogenize_points import homogenize_single_frame_points
from multiperson.utilities.display import draw_and_display_lines
from multiperson.utilities.video_io import get_synchronized_frames


def get_frames(cameras: list[Camera], active_frame: int, video_path: Path) -> list[np.ndarray]:
    """
    This will obviously look very different in practice
    """
    if active_frame == 500: # use cached values
        frames = [get_saved_frame(camera, active_frame) for camera in cameras]
    else:
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


if __name__ == "__main__":
    # path_to_calibration_toml = (
    #     Path(__file__).parent
    #     / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    # )

    path_to_calibration_toml = Path("/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/recording_14_30_34_gmt-6_calibration/recording_14_30_34_gmt-6_calibration_camera_calibration.toml")
    camera_collection = CameraCollection.from_file(path_to_calibration_toml)

    # video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/")
    video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/synchronized_videos/")

    id_list = camera_collection.ids
    a_index = 0
    b_index = 1

    camera_a = camera_collection.by_id(id_list[a_index])
    camera_b = camera_collection.by_id(id_list[b_index])

    fundamental = fundamental_from_camera_pair(camera_a, camera_b)

    # Get image points:
    # body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    body_data_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/output_data/raw_data/brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy")
    body_data_cams_frame_points_xy = np.load(body_data_path)

    # Everything above this only has to happen once per pair of cameras
    # Everything below this will have to happen per frame

    active_frame = 110  # use 500 for cached data
    number_of_tracked_points = 2

    image_a, image_b = get_frames([camera_a, camera_b], active_frame, video_path)

    image_a_points = homogenize_single_frame_points(
        body_data_cams_frame_points_xy[a_index, active_frame, :number_of_tracked_points, :2]
    )
    image_b_points = homogenize_single_frame_points(
        body_data_cams_frame_points_xy[b_index, active_frame, :number_of_tracked_points, :2]
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
