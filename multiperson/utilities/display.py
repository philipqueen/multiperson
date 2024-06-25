import cv2
import numpy as np


def display_frames(frames: dict[str, np.ndarray]):
    frame_group = tuple(frame for frame in frames.values())

    combined_frame = np.concatenate(frame_group, axis=1)

    cv2.imshow("Synchronized Frames", combined_frame)
    cv2.waitKey(0)

def draw_lines(image: np.ndarray, lines: np.ndarray, points: np.ndarray):
    """
    Draw points and epipolar lines on an image

    Corresponding points and lines are drawn in the same color
    Lines are represent in standard form, ax + by + c = 0, where a, b, and c are the first dimension of the output array.
    The shape of lines should be the transpose of the shape of points
    """
    annotated_image = image.copy()
    rows, columns = annotated_image.shape[:2]
    np.random.seed(0)
    for rows, point in zip(lines.T, points):
        # print(f"rows: {rows}, point: {point}")
        color = tuple(np.random.randint(0, 255, 3).tolist())
        try: # TODO: better way to deal with NaNs
            x0, y0 = map(int, [0, -rows[2] / rows[1]])
            x1, y1 = map(int, [columns, -(rows[2] + rows[0] * columns) / rows[1]])
            annotated_image = cv2.line(annotated_image, (x0, y0), (x1, y1), color, 1)
        except (ValueError, ZeroDivisionError):
            pass
        if not np.any(np.isnan(point[:2])):
            annotated_image = cv2.circle(annotated_image, tuple(point[:2].astype(int)), 5, color, -1)

    return annotated_image

def draw_and_display_lines(
    image_a: np.ndarray,
    image_b: np.ndarray,
    image_a_points: np.ndarray,
    image_b_points: np.ndarray,
    image_a_lines: np.ndarray,
    image_b_lines: np.ndarray,
):
    image_a_with_lines = draw_lines(image_a.copy(), image_a_lines, image_a_points)
    image_b_with_lines = draw_lines(image_b.copy(), image_b_lines, image_b_points)

    display_frames(
        frames={"image_a": image_a_with_lines, "image_b": image_b_with_lines}
    )