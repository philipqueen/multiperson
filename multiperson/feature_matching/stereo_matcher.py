from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiperson.data_models.camera_collection import CameraCollection
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
        homogenized_data_cams_frame_points_xy: np.ndarray,
        synchronized_video_folder_path: Union[str, Path],
        num_objects: int,
    ):
        self.camera_collection = camera_collection
        self.synchronized_video_folder_path = Path(synchronized_video_folder_path)
        self.num_objects = num_objects

        self._validate_input_index(index_a, index_b)
        self.index_a = index_a
        self.index_b = index_b

        self.data_cams_frame_points_xy = homogenized_data_cams_frame_points_xy

        self.camera_a = camera_collection.by_index(self.index_a)
        self.camera_b = camera_collection.by_index(self.index_b)

        self.fundamental = fundamental_from_camera_pair(self.camera_a, self.camera_b)

    def match_videos(self) -> None:
        """
        Function for running standalone StereoMatcher on pair of videos.
        Not to be used with NMatcher.
        """
        self._load_videos()
        video_writer = self._video_writer()

        for i in range(self.data_cams_frame_points_xy.shape[1]):
            swapped = False
            frame_a, frame_b = self._next_frames()
            current_points_a = self.data_cams_frame_points_xy[
                self.index_a, i, :, :
            ].copy()  # TODO: copies are probably redundant
            current_points_b = self.data_cams_frame_points_xy[
                self.index_b, i, :, :
            ].copy()
            image_b_lines = calculate_epipolar_lines(self.fundamental, current_points_a)
            image_a_lines = calculate_epipolar_lines(
                self.fundamental.T, current_points_b
            )

            cost_matrix = self._match_single_frame(
                current_points_a, current_points_b
            )

            current_ordering = self._order_by_costs_optimal(cost_matrix)

            matched_image_b_points = self.reorder_points(
                points=current_points_b, ordering=current_ordering
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

            original_a = draw_lines(
                frame_a,
                image_a_lines,
                current_points_a,
                points_per_object=points_per_object,
            )
            original_b = draw_lines(
                frame_b,
                image_b_lines,
                current_points_b,
                points_per_object=points_per_object,
            )
            matched_a = draw_lines(
                frame_a,
                matched_image_a_lines,
                current_points_a,
                points_per_object=points_per_object,
            )
            matched_b = draw_lines(
                frame_b,
                image_b_lines,
                matched_image_b_points,
                points_per_object=points_per_object,
            )

            self.add_frames(
                video_writer, [original_a, original_b, matched_a, matched_b], swapped
            )

        video_writer.release()

    def match_by_frame_number(self, frame_number: int) -> np.ndarray:
        """
        Public method to get cost matrix of objects in B by distance to the epipolar lines from objects in A for a given frame
        """
        points_a = self.data_cams_frame_points_xy[
            self.index_a, frame_number, :, :
        ].copy()  # TODO: copies are probably redundant
        points_b = self.data_cams_frame_points_xy[
            self.index_b, frame_number, :, :
        ].copy()

        return self._match_single_frame(points_a, points_b)

    def reorder_points(self, points: np.ndarray, ordering: List[int]) -> np.ndarray:
        """
        Given an ordering, reorder list of points by object and collapse back to single array of matched points.
        """

        points_by_object = np.split(points, self.num_objects, axis=0)

        reordered_points_by_object = [points_by_object[i] for i in ordering]

        matched_points = np.concatenate(reordered_points_by_object, axis=0)

        return matched_points

    def _match_single_frame(
        self, image_a_points: np.ndarray, image_b_points: np.ndarray
    ) -> np.ndarray:
        """
        Given points from two images, match greedily by minimizing distance to epipolar lines within each object

        Args:
            image_a_points: points in image_a
            image_b_points: points in image_b

        Returns:
            cost_matrix: matrix of objects in b by their distance to objects in a
        """
        image_b_lines = calculate_epipolar_lines(self.fundamental, image_a_points)

        # seperate points and lines by object
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
        return self._distances_to_cost_matrix(distances=point_to_lines_distances)

    def _order_by_costs_optimal(self, cost_matrix: np.ndarray) -> List[int]:
        """
        Make an ordering based on the cost matrix, which is distance between points with an artificial high value replacing NaNs.
        An ordering is a list of integers in the range [0, self.num_objects] mapping points to lines.
        Uses the "Hungarian algorithm" to find the optimal assignment.
        """
        _chosen_distances, assignments = linear_sum_assignment(cost_matrix)
        # total_distance = cost_matrix[chosen_distances, assignments].sum()  # computes the total distance between chosen points and lines
        return assignments.tolist()
    
    def _distances_to_cost_matrix(self, distances: np.ndarray) -> np.ndarray:
        # Replace NaN values with a high cost
        large_value = 1e10
        return np.nan_to_num(distances, nan=large_value)

    # TODO: all the video stuff should be in its own class
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
        video_path = self.synchronized_video_folder_path / f"{video_name}.mp4"

        return cv2.VideoCapture(str(video_path))

    def _video_writer(self) -> cv2.VideoWriter:
        save_path = Path(__file__).parent.parent.parent / f"stereo_match_{self.index_a}_{self.index_b}.mp4"
        print(f"will save output video to path {save_path}")
        return cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter.fourcc(*"mp4v"),
            30,
            (1440, 1080),
        )

    def add_frames(
        self,
        video_writer: cv2.VideoWriter,
        frame_list: List[
            np.ndarray
        ],  # TODO: get separate lists of before/after frames, and make sure they're display properly
        swapped: bool,
    ) -> None:
        if len(frame_list) % 2 != 0:
            print(
                f"WARNING: length of frame list is {len(frame_list)}, but must be even"
            )

        # combine 4 frames in a grid, with frames a and b on top, and frames c and d on bottom
        top_frame = np.concatenate(frame_list[0 : len(frame_list) // 2], axis=1)
        bottom_frame = np.concatenate(frame_list[len(frame_list) // 2 :], axis=1)
        frame = np.concatenate((top_frame, bottom_frame), axis=0)

        if swapped:
            frame = cv2.circle(
                frame, (frame.shape[1] // 2, frame.shape[0] // 2), 10, (0, 0, 255), -1
            )

        video_writer.write(cv2.resize(frame, (1440, 1080)))

    def _validate_input_index(self, index_a: int, index_b: int) -> None:
        if not {index_a, index_b}.issubset(set(self.camera_collection.indexes)):
            raise ValueError(
                f"Camera index must be in {self.camera_collection.indexes}"
            )


if __name__ == "__main__":
    # Brightest Point Definitions
    # path_to_calibration_toml = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/recording_14_30_34_gmt-6_calibration/recording_14_30_34_gmt-6_calibration_camera_calibration.toml"
    # )
    # points_per_object = 1
    # simple:
    # video_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/synchronized_videos/"
    # )
    # body_data_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/simple_test/output_data/raw_data/brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # )
    # complex:
    # video_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/complex_test/synchronized_videos/"
    # )
    # body_data_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/2_brightest_points_2_cams/complex_test/output_data/raw_data/brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # )

    # Sample Data
    # video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/")
    # body_data_path = "/Users/philipqueen/Documents/GitHub/multiperson/multiperson/assets/sample_data/2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # points_per_object = 533

    # Multiperson Definitions
    path_to_calibration_toml = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_07_36/recording_15_13_46_gmt-4_calibration/recording_15_13_46_gmt-4_calibration_camera_calibration.toml"
    )
    points_per_object = 17

    # # no contact:
    video_path = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact/synchronized_videos/"
    )
    body_data_path = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact/output_data/raw_data/yolo_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    )

    # # crossing behind:
    # video_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_23_37_gmt-4__multiperson_crossing_behind/synchronized_videos/"
    # )
    # body_data_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_23_37_gmt-4__multiperson_crossing_behind/output_data/raw_data/yolo2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # )

    # both moving:
    # video_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_25_13_gmt-4__multiperson_both_moving/synchronized_videos/"
    # )
    # body_data_path = Path(
    #     "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_25_13_gmt-4__multiperson_both_moving/output_data/raw_data/yolo2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    # )

    camera_collection = CameraCollection.from_file(path_to_calibration_toml)

    a_index = 0
    b_index = 1

    # Get image points:
    body_data_cams_frame_points_xy = homogenize_data_array(np.load(body_data_path))

    num_objects = body_data_cams_frame_points_xy.shape[2] // points_per_object

    matcher = StereoMatcher(
        camera_collection,
        a_index,
        b_index,
        body_data_cams_frame_points_xy,
        video_path,
        num_objects,
    )

    matcher.match_videos()
