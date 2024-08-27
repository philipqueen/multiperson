from pathlib import Path
from typing import Dict, List, Tuple, Union
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import numpy as np


from multiperson.data_models.camera_collection import CameraCollection
from multiperson.feature_matching.stereo_matcher import StereoMatcher
from multiperson.geometry.homogenize_points import homogenize_data_array


class NMatcher:
    def __init__(
        self,
        camera_collection: CameraCollection,
        data_cams_frame_points_xy: np.ndarray,
        synchronized_video_folder_path: Union[str, Path],
        points_per_object: int,
    ):
        self.camera_collection = camera_collection
        self.synchronized_video_folder_path = Path(synchronized_video_folder_path)
        self.points_per_object = points_per_object

        self._validate_input_data(data_cams_frame_points_xy)
        self.data_cams_frame_points_xy = homogenize_data_array(
            data_cams_frame_points_xy
        )

        self.num_objects = (
            self.data_cams_frame_points_xy.shape[2] // self.points_per_object
        )

        self.num_frames = self.data_cams_frame_points_xy.shape[1]

    def match(self):
        """
        Matches features across all cameras
        """
        # generate all valid camera pairs
        camera_pairs = [
            (pair[0].index, pair[1].index)
            for pair in combinations(self.camera_collection.cameras, 2)
        ]
        print(f"Number of cameras: {self.camera_collection.size}")
        print(f"Number of pairs: {len(camera_pairs)}")
        print(f"Pairs: {camera_pairs}")

        # setup list of stereo matchers for each pair
        pairs_to_matcher_map: Dict[Tuple[int, int], StereoMatcher] = {}
        for pair in camera_pairs:
            pairs_to_matcher_map[pair] = StereoMatcher(
                self.camera_collection,
                pair[0],
                pair[1],
                self.data_cams_frame_points_xy,
                self.synchronized_video_folder_path,
                self.num_objects,
            )

        for frame_number in range(self.num_frames):
            # construct our mega cost matrix
            # print(f"frame {frame_number}")
            # TODO: replace this array or arrays shenanigans with a big array, for broadcasting shenanigans
            array_of_cost_matrices = np.full(
                (self.camera_collection.size - 1, self.camera_collection.size - 1),
                fill_value=np.nan,
                dtype=object,
            )
            for pair, matcher in pairs_to_matcher_map.items():
                cost_matrix = matcher.match_by_frame_number(frame_number)
                array_of_cost_matrices[pair[0], pair[1] - 1] = cost_matrix
            if frame_number == 289:
                print(array_of_cost_matrices)

            # play funny hungarian algorithm game
            ordering = self._order_by_costs_optimal(array_of_cost_matrices[0, 0])
            ordering_list = [ordering]
            i = 0
            while i < (array_of_cost_matrices.shape[0] - 1):
                array_of_cost_matrices[i, i] = self._reorder_cost_matrix_columns(
                    cost_matrix=array_of_cost_matrices[i, i], ordering=ordering
                )

                i += 1
                for j in range(i, array_of_cost_matrices.shape[0]):
                    array_of_cost_matrices[i, j] = self._reorder_cost_matrix_rows(
                        cost_matrix=array_of_cost_matrices[i, j], ordering=ordering
                    )

                array_of_cost_matrices[0, i] = self._normalize_and_add_cost_matrices(
                    first_matrix=array_of_cost_matrices[0, i],
                    second_matrix=array_of_cost_matrices[i, i],
                )

                ordering = self._order_by_costs_optimal(array_of_cost_matrices[i, i])
                ordering_list.append(ordering)

            if frame_number == 289:
                print(array_of_cost_matrices[0, :])
                print(ordering_list)

            # rearrange data by ordering

    def _order_by_costs_optimal(self, cost_matrix: np.ndarray) -> List[int]:
        """
        Make an ordering based on the cost matrix, which is distance between points with an artificial high value replacing NaNs.
        An ordering is a list of integers in the range [0, self.num_objects] mapping points to lines.
        Uses the "Hungarian algorithm" to find the optimal assignment.
        """
        _chosen_distances, assignments = linear_sum_assignment(cost_matrix)
        # total_distance = cost_matrix[chosen_distances, assignments].sum()  # computes the total distance between chosen points and lines
        return assignments.tolist()

    def _normalize_and_add_cost_matrices(
        self, first_matrix: np.ndarray, second_matrix: np.ndarray
    ) -> np.ndarray:
        # TODO: normalize cost matrices
        return first_matrix + second_matrix

    def _reorder_cost_matrix_columns(
        self, cost_matrix: np.ndarray, ordering: List[int]
    ) -> np.ndarray:
        return cost_matrix[:, ordering]
    
    def _reorder_cost_matrix_rows(
        self, cost_matrix: np.ndarray, ordering: List[int]
    ) -> np.ndarray:
        return cost_matrix[ordering, :]

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
        "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact/output_data/raw_data/yolo2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
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

    matcher = NMatcher(
        camera_collection,
        body_data_cams_frame_points_xy,
        video_path,
        points_per_object,
    )

    matcher.match()
