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

    def match(self) -> np.ndarray:
        """
        Matches features across all cameras
        """
        # setup list of stereo matchers for each pair of cameras
        pair_to_stereo_matcher_map = self._create_stereo_matchers()

        for frame_number in range(self.num_frames):
            # construct our mega cost matrix
            # print(f"frame {frame_number}")
            total_cost_matrix_array = self._create_cost_matrix_array(
                pair_to_stereo_matcher_map, frame_number
            )

            # print(f"cost matrix: {total_cost_matrix_array}")

            # play funny hungarian algorithm game
            # TODO: this loop can be greatly improved, it's definitely not starting+ending at exactly the right place
            i = 0
            lower_bound = 0
            upper_bound = self.num_objects
            # print(f"bounds for i={i}: {lower_bound}, {upper_bound}")
            ordering = self._order_by_costs_optimal(
                total_cost_matrix_array[
                    lower_bound:upper_bound, lower_bound:upper_bound
                ]
            )
            ordering_list = [ordering]
            while i < (self.camera_collection.size - 2): # -1 for 0 indexing, -1 for size of matrix (1 less than number of cameras)
                total_cost_matrix_array[
                    lower_bound:upper_bound, lower_bound:upper_bound
                ] = self._reorder_cost_matrix_columns(
                    cost_matrix=total_cost_matrix_array[
                        lower_bound:upper_bound, lower_bound:upper_bound
                    ],
                    ordering=ordering,
                )

                i += 1
                lower_bound = i * self.num_objects
                upper_bound = lower_bound + self.num_objects
                # print(f"bounds for i={i}: {lower_bound}, {upper_bound}")

                # apply ordering across camera i's row, and add each reordered matrix to the corresponding matrix in camera 0's row
                for j in range(i, self.camera_collection.size - 1):
                    temp_lower = j * self.num_objects
                    temp_upper = temp_lower + self.num_objects
                    print(
                        f"temp bounds for j={j}: lower {temp_lower}, upper {temp_upper}"
                    )
                    total_cost_matrix_array[
                        lower_bound:upper_bound, temp_lower:temp_upper
                    ] = self._reorder_cost_matrix_rows(
                        cost_matrix=total_cost_matrix_array[
                            lower_bound:upper_bound, temp_lower:temp_upper
                        ],
                        ordering=ordering,
                    )
                    total_cost_matrix_array[
                        0 : self.num_objects, temp_lower:temp_upper
                    ] = self._normalize_and_add_cost_matrices(
                        first_matrix=total_cost_matrix_array[
                            0 : self.num_objects, temp_lower:temp_upper
                        ],
                        second_matrix=total_cost_matrix_array[
                            lower_bound:upper_bound, temp_lower:temp_upper
                        ],
                    )

                ordering = self._order_by_costs_optimal(
                    total_cost_matrix_array[
                        lower_bound:upper_bound, lower_bound:upper_bound
                    ]
                )
                # print(f"{ordering}")
                # if (
                #     len(ordering) > 0
                # ):  # TODO: figure out why these array changes added an empty list
                ordering_list.append(ordering)

            # print(f"cost matrix after: {total_cost_matrix_array[0 : self.num_objects, :]}")
            # print(f"ordering {ordering_list}")

            self._reorder_data_by_ordering_list(frame_number, ordering_list)

        return self.data_cams_frame_points_xy

    def _reorder_data_by_ordering_list(self, frame_number, ordering_list):
        for index, ordering in enumerate(ordering_list):
            index += 1  # we start reordering camera 1, so enumerate is off by 1
            self._reorder_points(
                    camera=index, frame=frame_number, ordering=ordering
                )

    def _create_cost_matrix_array(
        self,
        pair_to_stereo_matcher_map: Dict[Tuple[int, int], StereoMatcher],
        frame_number: int,
    ) -> np.ndarray:
        array_size = (self.camera_collection.size - 1) * self.num_objects
        cost_matrix_array = np.full((array_size, array_size), fill_value=np.nan)

        for pair, matcher in pair_to_stereo_matcher_map.items():
            cost_matrix = matcher.match_by_frame_number(frame_number)
            cost_matrix_array[
                pair[0] * self.num_objects : (pair[0] + 1) * self.num_objects,
                (pair[1] - 1) * self.num_objects : pair[1] * self.num_objects, # columns offset by 1 because first column is camera 1
            ] = cost_matrix

        return cost_matrix_array

        # array_of_cost_matrices = np.full(
        #     (self.camera_collection.size - 1, self.camera_collection.size - 1),
        #     fill_value=np.nan,
        #     dtype=object,
        # )

        # for pair, matcher in pair_to_stereo_matcher_map.items():
        #     cost_matrix = matcher.match_by_frame_number(frame_number)
        #     array_of_cost_matrices[pair[0], pair[1] - 1] = (
        #         cost_matrix  # columns offset by 1 because first column is camera 1
        #     )

        # return array_of_cost_matrices

    def _reorder_points(self, camera: int, frame: int, ordering: List[int]) -> None:
        """
        Given an ordering, reorder list of points by object and collapse back to single array of matched points.
        Operation happens in place.
        """

        points_by_object = np.split(
            self.data_cams_frame_points_xy[camera, frame, :], self.num_objects, axis=0
        )

        reordered_points_by_object = [points_by_object[i] for i in ordering]

        matched_points = np.concatenate(reordered_points_by_object, axis=0)

        self.data_cams_frame_points_xy[camera, frame, :] = matched_points

    def _generate_camera_pairs(self) -> List[Tuple[int, int]]:
        """
        Generate all valid pairs of camera indices with no duplicates
        Creates N * (N-1) / 2 pairs, where N is the number of cameras in the CameraCollection
        """
        return [
            (pair[0].index, pair[1].index)
            for pair in combinations(self.camera_collection.cameras, 2)
        ]

    def _create_stereo_matchers(self) -> Dict[Tuple[int, int], StereoMatcher]:
        """
        Create a StereoMatcher instance (value) for each unique camera pair (key, as a tuple), to be able to run stereo geometry calculations on.

        Uses the cameras in self.camera_collection to generate the pairs.
        """
        camera_pairs = self._generate_camera_pairs()

        pairs_to_matcher_map = {}
        for pair in camera_pairs:
            pairs_to_matcher_map[pair] = StereoMatcher(
                self.camera_collection,
                pair[0],
                pair[1],
                self.data_cams_frame_points_xy,
                self.synchronized_video_folder_path,
                self.num_objects,
            )

        return pairs_to_matcher_map

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
        # TODO: normalize cost matrices by min/max or z score normalization
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
        camera_collection=camera_collection,
        data_cams_frame_points_xy=body_data_cams_frame_points_xy,
        synchronized_video_folder_path=video_path,
        points_per_object=points_per_object,
    )

    matched_data = matcher.match()

    # save out matched_data
    for i in range((matched_data.shape[2] // points_per_object)):
        save_path = body_data_path.parent / (body_data_path.stem + f"_person{i}.npy")
        print(f"saving person {i} to {save_path}")

        np.save(save_path, matched_data[:, :, i * points_per_object : (i + 1) * points_per_object, :])