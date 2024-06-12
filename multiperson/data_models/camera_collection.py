from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from utilities.read_calibration_toml import read_calibration_toml


class Camera:
    def __init__(self, id: str, name: str, intrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
        self.id = id
        self.name = name
        self.intrinsic = intrinsic
        self.rotation = rotation
        self.translation = translation

class CameraCollection:
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras

    @property
    def ids(self):
        return [camera.id for camera in self.cameras]
    
    @property
    def names(self):
        return [camera.name for camera in self.cameras]

    @staticmethod
    def from_file(path_to_calibration_toml: Union[str, Path]):
        calibration = read_calibration_toml(path_to_calibration_toml)

        return CameraCollection.from_dict(calibration)

    @staticmethod
    def from_dict(calibration: dict[str, dict[str, Any]]):
        return CameraCollection(
            [
                Camera(
                    id=key,
                    name=calibration[key]["name"],
                    intrinsic=calibration[key]["instrinsics_matrix"],
                    rotation=calibration[key]["rotation"],
                    translation=calibration[key]["translation"],
                )
                for key in calibration
            ]
        )
    
    def by_id(self, camera_id: str) -> Camera:
        for camera in self.cameras:
            if camera.id == camera_id:  # ids are guaranteed to be unique IF class is initialized from a dict
                return camera
        raise ValueError(f"Camera with id {camera_id} not found in CameraCollection")
