from typing import Union
import numpy as np
from scipy.optimize import linear_sum_assignment

class ExampleClass:
    def __init__(self, num_objects):
        self.num_objects = num_objects

    # def _order_by_distances(self, distances: Union[list[list[float]], np.ndarray]) -> list[int]:
    #     """
    #     Make an ordering based on the distances between points.
    #     An ordering is a list of integers in the range [0, self.num_objects) mapping points to lines.
    #     Each point is mapped to the line with the smallest distance, but each line is only mapped to one point.
    #     """
    #     distances = np.array(distances)
    #     row_ind, col_ind = linear_sum_assignment(distances)
    #     print(row_ind.tolist(), col_ind.tolist())
    #     return col_ind.tolist()
    
    def _order_by_distances(self, distances: list[list[float]]) -> list[int]:
        """
        Make an ordering based on the distances between points.
        An ordering is a list of integers in the range [0, self.num_objects) mapping points to lines.
        Each point is mapped to the line with the smallest distance, but each line is only mapped to one point.
        """
        # Currently, this is AI magic that works on simple test cases, but needs testing and optimization
        # TODO: look into assignment problem in combinatorics

        # Flatten the distances and sort by distance value
        flat_distances = [(i, j, distances[i][j]) for i in range(self.num_objects) for j in range(self.num_objects)]
        flat_distances.sort(key=lambda x: x[2])

        selected_points = set()
        selected_lines = set()
        ordering = [-1] * self.num_objects

        for point, line, distance in flat_distances:
            if point not in selected_points and line not in selected_lines:
                ordering[point] = line
                selected_points.add(point)
                selected_lines.add(line)

            if len(selected_points) == self.num_objects:
                break

        return ordering

# Example usage:
example = ExampleClass(num_objects=3)
distances = [
    [1.2, 2.3, 3],
    [4.5, 2, 0.9],
    [3.0, 2.2, 1.5]
]
# Expected output: [0, 2, 1]
ordering = example._order_by_distances(distances)
print(ordering) 