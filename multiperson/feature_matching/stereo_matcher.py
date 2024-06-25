from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiperson.data_models.camera_collection import Camera, CameraCollection
from multiperson.geometry.calculate_distance_to_lines import calculate_distance_to_lines
from multiperson.geometry.epipolar_geometry import (
    calculate_epipolar_lines,
    fundamental_from_camera_pair,
)
from multiperson.geometry.homogenize_points import (
    homogenize_data_array,
)
from multiperson.utilities.display import draw_lines


class StereoMatcher:
    def __init__(
        self,
        camera_collection: CameraCollection,
        index_a: int,
        index_b: int,
        data_cams_frame_points_xy: np.ndarray,
        synchronized_video_path: Union[str, Path],
        points_per_object: int,
    ):
        self.camera_collection = camera_collection
        self.synchronized_video_path = Path(synchronized_video_path)
        self.points_per_object = points_per_object

        self._validate_input_index(index_a, index_b)
        self.index_a = index_a
        self.index_b = index_b

        self._validate_input_data(data_cams_frame_points_xy)
        self.data_cams_frame_points_xy = homogenize_data_array(
            data_cams_frame_points_xy
        )
        self.num_objects = (
            self.data_cams_frame_points_xy.shape[2] // self.points_per_object
        )

        self.camera_a = camera_collection.by_index(self.index_a)
        self.camera_b = camera_collection.by_index(self.index_b)

        self.fundamental = fundamental_from_camera_pair(self.camera_a, self.camera_b)

    def match_videos(self) -> None:
        self._load_videos()
        video_writer = self._video_writer()

        for i in range(self.data_cams_frame_points_xy.shape[1]):
            swapped = False
            frame_a, frame_b = self._next_frames()
            current_points_a = self.data_cams_frame_points_xy[
                a_index, i, :, :
            ].copy()  # TODO: these are probably redundant
            current_points_b = self.data_cams_frame_points_xy[b_index, i, :, :].copy()
            image_b_lines = calculate_epipolar_lines(self.fundamental, current_points_a)
            image_a_lines = calculate_epipolar_lines(
                self.fundamental.T, current_points_b
            )

            matched_image_b_points = self._match_single_frame(
                current_points_a, current_points_b
            )

            if not np.all(
                np.isclose(
                    self.data_cams_frame_points_xy[b_index, i, :, :],
                    matched_image_b_points,
                )
            ):
                swapped = True
                self.data_cams_frame_points_xy[b_index, i, :, :] = (
                    matched_image_b_points
                )

            matched_image_a_lines = calculate_epipolar_lines(
                self.fundamental.T, matched_image_b_points
            )

            original_a = draw_lines(frame_a, image_a_lines, current_points_a)
            original_b = draw_lines(frame_b, image_b_lines, current_points_b)
            matched_a = draw_lines(frame_a, matched_image_a_lines, current_points_a)
            matched_b = draw_lines(frame_b, image_b_lines, matched_image_b_points)

            self.add_frames(
                video_writer, original_a, original_b, matched_a, matched_b, swapped
            )

        video_writer.release()

    def _match_single_frame(
        self, image_a_points: np.ndarray, image_b_points: np.ndarray
    ) -> np.ndarray:
        """
        Given points from two images, match greedily by minimizing distance to epipolar lines within each object

        Args:
            image_a_points: points in image_a
            image_b_points: points in image_b

        Returns:
            matched_image_b_points: reordered points in image_b based on greedy matching
        """
        image_b_lines = calculate_epipolar_lines(self.fundamental, image_a_points)

        image_b_points_by_object = np.split(image_b_points, self.num_objects, axis=0)
        image_b_lines_by_object = np.split(image_b_lines, self.num_objects, axis=1)

        point_to_lines_distances = np.full((self.num_objects, self.num_objects), np.nan)
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                point_to_lines_distances[i, j] = np.nanmean(
                    calculate_distance_to_lines(
                        image_b_points_by_object[i], image_b_lines_by_object[j]
                    )
                )
        ordering = self._order_by_distances_optimal(point_to_lines_distances)

        matched_image_b_points = np.take(image_b_points, ordering, axis=0)

        return matched_image_b_points

    def _order_by_distances_optimal(self, distances: np.ndarray) -> List[int]:
        """
        Make an ordering based on the distances between points.
        An ordering is a list of integers in the range [0, self.num_objects) mapping points to lines.
        Uses the "Hungarian algorithm" to find the optimal assignment.
        """
        # Replace NaN values with a high cost
        large_value = 1e10
        cost_matrix = np.nan_to_num(distances, nan=large_value)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind.tolist()

    def _next_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        reta, image_a = self.video_a.read()
        retb, image_b = self.video_b.read()

        if not reta or not retb:
            raise ValueError(
                "Could not load next frame, ensure length of videos matches length of data"
            )

        return image_a, image_b

    def _load_videos(self) -> None:
        self.video_a = self._load_video(self.index_a)
        self.video_b = self._load_video(self.index_b)

    def _load_video(self, index: int) -> cv2.VideoCapture:
        video_name = self.camera_collection.by_index(index).name
        video_path = self.synchronized_video_path / f"{video_name}.mp4"

        return cv2.VideoCapture(str(video_path))

    def _video_writer(self) -> cv2.VideoWriter:
        return cv2.VideoWriter(
            str(
                Path(__file__).parent.parent.parent
                / f"stereo_match_{self.index_a}_{self.index_b}.mp4"
            ),
            cv2.VideoWriter.fourcc(*"mp4v"),
            30,
            (1440, 1080),
        )

    def add_frames(
        self,
        video_writer: cv2.VideoWriter,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        frame_c: np.ndarray,
        frame_d: np.ndarray,
        swapped: bool,
    ) -> None:
        # combine 4 frames in a grid, with frames a and b on top, and frames c and d on bottom
        top_frame = np.concatenate((frame_a, frame_b), axis=1)
        bottom_frame = np.concatenate((frame_c, frame_d), axis=1)
        frame = np.concatenate((top_frame, bottom_frame), axis=0)

        if swapped:
            frame = cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 10, (0, 0, 255), -1)

        video_writer.write(cv2.resize(frame, (1440, 1080)))

    def _validate_input_index(self, index_a: int, index_b: int) -> None:
        if not {index_a, index_b}.issubset(set(self.camera_collection.indexes)):
            raise ValueError(
                f"Camera index must be in {self.camera_collection.indexes}"
            )

    def _validate_input_data(self, data_cams_frame_points_xy: np.ndarray) -> None:
        if data_cams_frame_points_xy.ndim != 4:
            raise ValueError(
                "Data must be 4D (num_cams, num_frames, tracked_points, xy/xyz)"
            )
        if data_cams_frame_points_xy.shape[0] != len(self.camera_collection.ids):
            raise ValueError(
                "Number of cameras in camera collection must match number of cameras in data"
            )
        if data_cams_frame_points_xy.shape[2] % self.points_per_object != 0:
            raise ValueError(
                "Number of tracked points must be a multiple of points_per_object"
            )
        if data_cams_frame_points_xy.shape[3] not in {2, 3}:
            raise ValueError("Data must be 2D (N, 2) or 3D (N, 3)")


if __name__ == "__main__":
    path_to_calibration_toml = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/recording_14_30_34_gmt-6_calibration/recording_14_30_34_gmt-6_calibration_camera_calibration.toml"
    )
    camera_collection = CameraCollection.from_file(path_to_calibration_toml)

    # video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/")
    video_path = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/synchronized_videos/"
    )
    # video_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/complex_test/synchronized_videos/"
    # )

    a_index = 0
    b_index = 1

    # points_per_object = 533
    points_per_object = 1

    # Get image points:
    # body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    body_data_path = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/output_data/raw_data/brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    )
    # body_data_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/complex_test/output_data/raw_data/brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # )
    body_data_cams_frame_points_xy = np.load(body_data_path)

    matcher = StereoMatcher(
        camera_collection,
        a_index,
        b_index,
        body_data_cams_frame_points_xy,
        video_path,
        points_per_object,
    )

    matcher.match_videos()
