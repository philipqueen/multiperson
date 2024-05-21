import cv2
import numpy as np
from pathlib import Path

from multiperson.utilities.get_video_paths import get_video_paths

def get_synchronized_frames(
    video_folder_path: str | Path,
    frame_number: int = 200
) -> dict[str, np.ndarray]:
    video_folder_path = Path(video_folder_path)

    video_paths = get_video_paths(video_folder_path)

    frames = {}
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_number > video_length:
            print(f"WARNING: frame_number ({frame_number}) > video_length ({video_length})")
            print("Setting frame_number = video_length")
            frame_number = video_length

        for _ in range(frame_number):
            ret, frame = cap.read()
        
        frames[str(video_path)] = frame

        cap.release()

    return frames

def display_frames(frames: dict[str, np.ndarray]):
    frame_group = tuple(frame for frame in frames.values())

    combined_frame = np.concatenate(frame_group, axis=1)

    cv2.imshow("Synchronized Frames", combined_frame)
    cv2.waitKey(0)

def save_frames(frames: dict[str, np.ndarray], frame_number: int):
    for frame_path, frame in frames.items():
        name = Path(frame_path).stem + f"_{frame_number}" + ".jpg"
        save_folder = Path(__file__).parent.parent / "assets/synchronized_frames"

        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        save_path = save_folder / name
        print(f"Saving frame to: {save_path}")
        cv2.imwrite(str(save_path), frame)

if __name__ == "__main__":
    video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/")
    frame_number = 170

    frames = get_synchronized_frames(video_path, frame_number)

    display_frames(frames)

    save_frames(frames, frame_number)