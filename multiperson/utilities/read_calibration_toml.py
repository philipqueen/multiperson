from typing import Any
import rtoml
from pathlib import Path

def read_calibration_toml(path_to_calibration_toml: str | Path) -> dict[str, dict[str, Any]]:
    path_to_calibration_toml = Path(path_to_calibration_toml)
    calibration = rtoml.load(path_to_calibration_toml)

    processed_calibration = {}
    for key, value in calibration.items():
        if key != "metadata":
            processed_calibration[key] = {}
            processed_calibration[key]["name"] = value["name"]
            processed_calibration[key]["instrinsics_matrix"] = value["matrix"]
            processed_calibration[key]["rotation"] = value["rotation"]
            processed_calibration[key]["translation"] = value["translation"]

    return processed_calibration

if __name__ == "__main__":
    path_to_calibration_toml = Path(__file__).parent.parent / "assets/sample_data/freemocap_sample_data_camera_calibration.toml"
    calibration = read_calibration_toml(path_to_calibration_toml)

    print(calibration)