from typing import Any
import cv2
import numpy as np
import rtoml
from pathlib import Path

def read_calibration_toml(path_to_calibration_toml: str | Path) -> dict[str, dict[str, Any]]:
    path_to_calibration_toml = Path(path_to_calibration_toml)
    calibration = rtoml.load(path_to_calibration_toml)

    processed_calibration = {}
    for index, (key, value) in enumerate(calibration.items()):
        if key != "metadata":
            processed_calibration[key] = {}
            processed_calibration[key]["index"] = index  # this is better than taking the suffix of the key, since the user could manually edit the calibration toml
            processed_calibration[key]["name"] = value["name"]
            processed_calibration[key]["instrinsics_matrix"] = np.array(value["matrix"])
            processed_calibration[key]["rotation"], _ = cv2.Rodrigues(np.array(value["rotation"]))
            processed_calibration[key]["translation"] = np.array(value["translation"])

    return processed_calibration

if __name__ == "__main__":
    path_to_calibration_toml = Path(__file__).parent.parent / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    calibration = read_calibration_toml(path_to_calibration_toml)

    print(calibration)