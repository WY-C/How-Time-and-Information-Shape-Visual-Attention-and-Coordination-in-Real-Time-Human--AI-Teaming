from collections import deque
import ctypes
import ctypes.wintypes
from dataclasses import dataclass
import json
import os
import threading
import time

import numpy as np
import pygame

from ivt import IVTConfig, angular_distance_deg


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GAZE_CALIBRATION_POINTS = 9
DEFAULT_GAZE_CAPTURE_WINDOW_SEC = 0.35
DEFAULT_GAZE_CAPTURE_MIN_SAMPLES = 8
DEFAULT_SHARED_GAZE_CALIBRATION_FILENAME = "shared.json"
GAZE_CALIBRATION_FORMAT_VERSION = 3
GAZE_CALIBRATION_SCOPE = "display"
GAZE_CALIBRATION_TARGETS = (
    (0.50, 0.50),
    (0.15, 0.15),
    (0.85, 0.85),
    (0.85, 0.15),
    (0.15, 0.85),
    (0.50, 0.15),
    (0.85, 0.50),
    (0.50, 0.85),
    (0.15, 0.50),
)
DEFAULT_VIEWING_DISTANCE_WINDOW_SEC = 2.0
DEFAULT_VIEWING_DISTANCE_MIN_SAMPLES = 30
CALIBRATION_WARNING_RMSE = 0.03
CALIBRATION_MAX_RMSE = 0.05
CALIBRATION_MIN_TARGET_SPAN = 0.5
DRIFT_CHECK_DURATION_SEC = 1.0
DRIFT_CHECK_MIN_SAMPLES = 30
DRIFT_CHECK_MAX_DEGREES = 2.0
GAZE_PREVIEW_WINDOW_SEC = 0.5
GAZE_PREVIEW_MIN_SAMPLES = 15
GAZE_PREVIEW_TARGETS = (
    ("PLATE", 0, 0, os.path.join("lib", "overcooked_ai", "overcooked_ai_js", "assets", "objects", "dish.png")),
    (
        "CHEF",
        5,
        0,
        None,
    ),
    ("POT", 2, 2, os.path.join("lib", "overcooked_ai", "overcooked_ai_js", "assets", "terrain", "pot.png")),
    (
        "SERVING",
        0,
        3,
        os.path.join("lib", "overcooked_ai", "overcooked_ai_js", "assets", "terrain", "serve.png"),
    ),
    (
        "ONION",
        5,
        3,
        os.path.join("lib", "overcooked_ai", "overcooked_ai_js", "assets", "terrain", "onions.png"),
    ),
)
_GAZE_PREVIEW_IMAGE_CACHE = {}


@dataclass(frozen=True)
class TobiiTimebase:
    """Map Tobii system timestamps onto Python's monotonic clock."""

    system_timestamp_us: int
    perf_counter: float

    def to_perf_counter(self, system_timestamp_us):
        if system_timestamp_us is None:
            return None
        return self.perf_counter + (
            int(system_timestamp_us) - self.system_timestamp_us
        ) / 1_000_000.0


def capture_tobii_timebase(get_system_timestamp, perf_counter=time.perf_counter):
    """Capture both clocks closely enough to timestamp later gaze samples."""

    before = perf_counter()
    system_timestamp_us = int(get_system_timestamp())
    after = perf_counter()
    return TobiiTimebase(
        system_timestamp_us=system_timestamp_us,
        perf_counter=(before + after) / 2.0,
    )


def extract_tobii_gaze_origins(gaze_data):
    origins = {}
    for eye_prefix in ("left", "right"):
        point = gaze_data.get(f"{eye_prefix}_gaze_origin_in_user_coordinate_system")
        validity = gaze_data.get(f"{eye_prefix}_gaze_origin_validity")
        valid = validity is None or bool(validity)
        if point is None:
            valid = False
            normalized_point = None
        else:
            normalized_point = tuple(float(value) for value in point[:3])
            valid = valid and len(normalized_point) == 3 and all(
                np.isfinite(value) for value in normalized_point
            )
        origins[eye_prefix] = {
            "valid": bool(valid),
            "point": normalized_point,
        }
    return origins


class TobiiViewingDistanceEstimator:
    """Estimate eye-to-display-plane distance from recent Tobii gaze origins."""

    def __init__(self, display_area, minimum_mm=250.0, maximum_mm=1200.0):
        self.minimum_mm = float(minimum_mm)
        self.maximum_mm = float(maximum_mm)
        self._samples = deque(maxlen=4096)
        self._sample_lock = threading.Lock()
        self._plane_point = None
        self._plane_normal = None
        if display_area is not None:
            self._set_display_plane(display_area)

    @staticmethod
    def _display_point(display_area, name):
        if isinstance(display_area, dict):
            return display_area.get(name)
        return getattr(display_area, name, None)

    def _set_display_plane(self, display_area):
        top_left = self._display_point(display_area, "top_left")
        top_right = self._display_point(display_area, "top_right")
        bottom_left = self._display_point(display_area, "bottom_left")
        if top_left is None or top_right is None or bottom_left is None:
            return
        top_left = np.asarray(top_left, dtype=np.float64)
        top_right = np.asarray(top_right, dtype=np.float64)
        bottom_left = np.asarray(bottom_left, dtype=np.float64)
        if any(point.shape != (3,) for point in (top_left, top_right, bottom_left)):
            return
        normal = np.cross(top_right - top_left, bottom_left - top_left)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm <= 0:
            return
        self._plane_point = top_left
        self._plane_normal = normal / normal_norm

    @property
    def has_display_plane(self):
        return self._plane_point is not None and self._plane_normal is not None

    def add_gaze_data(self, gaze_data, timestamp):
        if not self.has_display_plane:
            return
        distances = []
        for origin in extract_tobii_gaze_origins(gaze_data).values():
            if not origin["valid"]:
                continue
            point = np.asarray(origin["point"], dtype=np.float64)
            distance = abs(float(np.dot(point - self._plane_point, self._plane_normal)))
            if self.minimum_mm <= distance <= self.maximum_mm:
                distances.append(distance)
        if distances:
            with self._sample_lock:
                self._samples.append((float(timestamp), float(np.mean(distances))))

    def estimate(
        self,
        fallback_mm,
        *,
        now=None,
        window_sec=DEFAULT_VIEWING_DISTANCE_WINDOW_SEC,
        minimum_samples=DEFAULT_VIEWING_DISTANCE_MIN_SAMPLES,
    ):
        now = time.perf_counter() if now is None else float(now)
        cutoff = now - float(window_sec)
        with self._sample_lock:
            recent_samples = list(self._samples)
        values = np.asarray(
            [distance for timestamp, distance in recent_samples if timestamp >= cutoff],
            dtype=np.float64,
        )
        enough_samples = len(values) >= int(minimum_samples)
        applied_distance = float(np.median(values)) if enough_samples else float(fallback_mm)
        return {
            "source": "tobii_gaze_origin" if enough_samples else "configured_fallback",
            "applied_distance_mm": applied_distance,
            "sample_count": int(len(values)),
            "mean_mm": None if len(values) == 0 else float(np.mean(values)),
            "median_mm": None if len(values) == 0 else float(np.median(values)),
            "std_mm": None if len(values) == 0 else float(np.std(values)),
            "window_sec": float(window_sec),
        }


def evaluate_gaze_drift(
    raw_display_points,
    gaze_calibration,
    viewing_distance_mm,
    *,
    screen_diagonal_inches=27.0,
    target_display_point=(0.5, 0.5),
    minimum_samples=DRIFT_CHECK_MIN_SAMPLES,
    maximum_degrees=DRIFT_CHECK_MAX_DEGREES,
):
    corrected_points = []
    for point in raw_display_points:
        corrected = gaze_calibration.apply(point)
        if corrected is None or not all(np.isfinite(value) for value in corrected):
            continue
        corrected_points.append(corrected)

    result = {
        "passed": False,
        "sample_count": len(corrected_points),
        "angle_deg": None,
        "centroid": None,
        "target": [float(target_display_point[0]), float(target_display_point[1])],
        "threshold_deg": float(maximum_degrees),
    }
    if len(corrected_points) < int(minimum_samples):
        result["status"] = "insufficient_samples"
        return result

    centroid = tuple(float(np.median([point[index] for point in corrected_points])) for index in range(2))
    config = IVTConfig(
        screen_diagonal_inches=float(screen_diagonal_inches),
        viewing_distance_mm=float(viewing_distance_mm),
    )
    angle_deg = angular_distance_deg(centroid, target_display_point, config)
    result.update(
        {
            "passed": bool(angle_deg <= maximum_degrees),
            "status": "pass" if angle_deg <= maximum_degrees else "fail",
            "angle_deg": float(angle_deg),
            "centroid": list(centroid),
        }
    )
    return result


class TobiiGazeTracker:
    def __init__(self, requested_address=None):
        try:
            import tobii_research as tr
        except ImportError as exc:
            raise ImportError(
                "tobii_research is required for --show-tobii-gaze, but it is not installed in the current Python environment."
            ) from exc

        eye_trackers = tr.find_all_eyetrackers()
        if not eye_trackers:
            raise RuntimeError("No Tobii eye tracker was detected.")

        if requested_address is None:
            self.eye_tracker = eye_trackers[0]
        else:
            requested_value = requested_address.strip()
            self.eye_tracker = next(
                (
                    eye_tracker
                    for eye_tracker in eye_trackers
                    if requested_value
                    in {getattr(eye_tracker, "address", ""), getattr(eye_tracker, "serial_number", "")}
                ),
                None,
            )
            if self.eye_tracker is None:
                available_trackers = [
                    {
                        "address": getattr(eye_tracker, "address", ""),
                        "serial_number": getattr(eye_tracker, "serial_number", ""),
                        "model": getattr(eye_tracker, "model", ""),
                    }
                    for eye_tracker in eye_trackers
                ]
                raise RuntimeError(
                    f"Tobii eye tracker '{requested_value}' was not found. Available trackers: {available_trackers}"
                )

        self._tr = tr
        self.latest_gaze_point = None
        self._recent_gaze_points = deque(maxlen=512)
        self._recent_gaze_lock = threading.Lock()
        self._raw_gaze_samples = deque(maxlen=4096)
        self._raw_sample_lock = threading.Lock()
        self._timebase = capture_tobii_timebase(self._tr.get_system_time_stamp)
        try:
            display_area = self.eye_tracker.get_display_area()
        except Exception as exc:
            display_area = None
            print(f"[Tobii] Display area unavailable; viewing distance will use fallback: {exc}")
        self._viewing_distance_estimator = TobiiViewingDistanceEstimator(display_area)
        self._subscribed = False
        self._subscribe()
        print(
            "[Tobii] Connected "
            f"model={getattr(self.eye_tracker, 'model', '')} "
            f"serial={getattr(self.eye_tracker, 'serial_number', '')} "
            f"address={getattr(self.eye_tracker, 'address', '')}"
        )

    def _extract_valid_points(self, gaze_data):
        valid_points = []
        for eye_prefix in ("left", "right"):
            point = gaze_data.get(f"{eye_prefix}_gaze_point_on_display_area")
            validity = gaze_data.get(f"{eye_prefix}_gaze_point_validity")
            if point is None:
                continue
            if validity is None or validity == 1:
                x, y = float(point[0]), float(point[1])
                if np.isfinite(x) and np.isfinite(y):
                    valid_points.append((x, y))
        return valid_points

    def _handle_gaze_data(self, gaze_data):
        valid_points = self._extract_valid_points(gaze_data)
        gaze_point_validities = {
            eye_prefix: (
                None
                if gaze_data.get(f"{eye_prefix}_gaze_point_validity") is None
                else bool(gaze_data[f"{eye_prefix}_gaze_point_validity"])
            )
            for eye_prefix in ("left", "right")
        }
        callback_timestamp = time.perf_counter()
        system_timestamp_us = gaze_data.get("system_time_stamp")
        sample_timestamp = self._timebase.to_perf_counter(system_timestamp_us)
        if sample_timestamp is None:
            sample_timestamp = callback_timestamp
        gaze_origins = extract_tobii_gaze_origins(gaze_data)
        self._viewing_distance_estimator.add_gaze_data(gaze_data, sample_timestamp)
        if not valid_points:
            with self._recent_gaze_lock:
                self.latest_gaze_point = None
            with self._raw_sample_lock:
                self._raw_gaze_samples.append(
                    {
                        "perf_counter": sample_timestamp,
                        "callback_perf_counter": callback_timestamp,
                        "point": None,
                        "valid": False,
                        "device_timestamp_us": gaze_data.get("device_time_stamp"),
                        "system_timestamp_us": system_timestamp_us,
                        "gaze_point_validities": gaze_point_validities,
                        "gaze_origins": gaze_origins,
                    }
                )
            return
        mean_x = float(sum(point[0] for point in valid_points) / len(valid_points))
        mean_y = float(sum(point[1] for point in valid_points) / len(valid_points))
        gaze_point = (mean_x, mean_y)
        with self._recent_gaze_lock:
            self.latest_gaze_point = gaze_point
            self._recent_gaze_points.append((sample_timestamp, gaze_point))
        with self._raw_sample_lock:
            self._raw_gaze_samples.append(
                {
                    "perf_counter": sample_timestamp,
                    "callback_perf_counter": callback_timestamp,
                    "point": gaze_point,
                    "valid": True,
                    "device_timestamp_us": gaze_data.get("device_time_stamp"),
                    "system_timestamp_us": system_timestamp_us,
                    "gaze_point_validities": gaze_point_validities,
                    "gaze_origins": gaze_origins,
                }
            )

    def estimate_viewing_distance(self, fallback_mm):
        return self._viewing_distance_estimator.estimate(fallback_mm)

    def get_recent_average_gaze_point(
        self,
        max_age_sec=DEFAULT_GAZE_CAPTURE_WINDOW_SEC,
        min_samples=DEFAULT_GAZE_CAPTURE_MIN_SAMPLES,
    ):
        cutoff_time = time.perf_counter() - max_age_sec
        with self._recent_gaze_lock:
            recent_points = [
                point
                for timestamp, point in self._recent_gaze_points
                if timestamp >= cutoff_time
            ]
        if len(recent_points) < min_samples:
            return None
        mean_x = float(sum(point[0] for point in recent_points) / len(recent_points))
        mean_y = float(sum(point[1] for point in recent_points) / len(recent_points))
        return mean_x, mean_y

    def get_recent_valid_gaze_sample_count(self, max_age_sec=GAZE_PREVIEW_WINDOW_SEC):
        cutoff_time = time.perf_counter() - float(max_age_sec)
        with self._recent_gaze_lock:
            return sum(
                1
                for timestamp, _ in self._recent_gaze_points
                if timestamp >= cutoff_time
            )

    def get_recent_valid_gaze_points(self, max_age_sec=DRIFT_CHECK_DURATION_SEC):
        cutoff_time = time.perf_counter() - float(max_age_sec)
        with self._recent_gaze_lock:
            return [
                point
                for timestamp, point in self._recent_gaze_points
                if timestamp >= cutoff_time
            ]

    def clear_recent_gaze_points(self):
        with self._recent_gaze_lock:
            self._recent_gaze_points.clear()
            self.latest_gaze_point = None

    def _subscribe(self):
        self.eye_tracker.subscribe_to(
            self._tr.EYETRACKER_GAZE_DATA,
            self._handle_gaze_data,
            as_dictionary=True,
        )
        self._subscribed = True

    def drain_gaze_samples(self):
        with self._raw_sample_lock:
            samples = list(self._raw_gaze_samples)
            self._raw_gaze_samples.clear()
        return samples

    def close(self):
        if not self._subscribed:
            return
        self.eye_tracker.unsubscribe_from(
            self._tr.EYETRACKER_GAZE_DATA,
            self._handle_gaze_data,
        )
        self._subscribed = False


class GazeCalibration:
    def __init__(
        self,
        layout_name,
        calibration_path=None,
        required_points=DEFAULT_GAZE_CALIBRATION_POINTS,
        participant_id=None,
    ):
        self.layout_name = layout_name
        self.format_version = GAZE_CALIBRATION_FORMAT_VERSION
        self.calibration_scope = GAZE_CALIBRATION_SCOPE
        if int(required_points) != len(GAZE_CALIBRATION_TARGETS):
            raise ValueError(
                f"Display calibration requires exactly {len(GAZE_CALIBRATION_TARGETS)} points."
            )
        self.required_points = len(GAZE_CALIBRATION_TARGETS)
        self.participant_id = None if participant_id is None else str(participant_id).strip().upper()
        self._calibration_dir = os.path.abspath(
            os.path.join(PROJECT_ROOT, "data", "raw", "gaze_calibration")
        )
        self._explicit_calibration_path = os.path.abspath(calibration_path) if calibration_path else None
        calibration_file_name = DEFAULT_SHARED_GAZE_CALIBRATION_FILENAME
        if self.participant_id:
            safe_participant_id = "".join(
                character if character.isalnum() or character in {"-", "_"} else "_"
                for character in self.participant_id
            )
            calibration_file_name = f"{safe_participant_id}.json"
        self.calibration_path = self._explicit_calibration_path or os.path.join(
            self._calibration_dir,
            calibration_file_name,
        )
        self.shared_calibration_path = self.calibration_path
        self.loaded_from_path = None
        self.transform_matrix = None
        self.fit_rmse = None
        self.samples = []
        self.is_collecting = False
        self.status_message = "Press F6 to start participant display calibration."

    def _get_candidate_load_paths(self):
        return [self.calibration_path]

    def _get_candidate_save_paths(self):
        return [self.calibration_path]

    def _build_payload(self):
        return {
            "version": GAZE_CALIBRATION_FORMAT_VERSION,
            "calibration_scope": GAZE_CALIBRATION_SCOPE,
            "participant_id": self.participant_id,
            "source_layout_name": self.layout_name,
            "required_points": self.required_points,
            "fit_rmse": self.fit_rmse,
            "transform_matrix": self.transform_matrix.tolist(),
            "samples": self.samples,
        }

    def _write_payload(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._build_payload(), handle, indent=2)

    @property
    def sample_count(self):
        return len(self.samples)

    @property
    def has_transform(self):
        return self.transform_matrix is not None

    @property
    def current_target_normalized(self):
        if not self.is_collecting or self.sample_count >= self.required_points:
            return None
        return GAZE_CALIBRATION_TARGETS[self.sample_count]

    def current_target_window_point(self, window_size):
        target = self.current_target_normalized
        if target is None:
            return None
        return target[0] * float(window_size[0]), target[1] * float(window_size[1])

    @property
    def target_span(self):
        target_points = [sample.get("window_normalized_point") for sample in self.samples]
        if not any(point is not None for point in target_points):
            target_points = [sample.get("target_display_point") for sample in self.samples]
        target_points = [point for point in target_points if point is not None and len(point) >= 2]
        if not target_points:
            return 0.0, 0.0
        x_values = [float(point[0]) for point in target_points]
        y_values = [float(point[1]) for point in target_points]
        return max(x_values) - min(x_values), max(y_values) - min(y_values)

    @property
    def quality_status(self):
        if not self.has_transform or not isinstance(self.fit_rmse, (float, int)):
            return "unavailable"
        if self.sample_count < self.required_points:
            return "insufficient_points"
        span_x, span_y = self.target_span
        if span_x < CALIBRATION_MIN_TARGET_SPAN or span_y < CALIBRATION_MIN_TARGET_SPAN:
            return "insufficient_coverage"
        if self.fit_rmse <= CALIBRATION_WARNING_RMSE:
            return "pass"
        if self.fit_rmse <= CALIBRATION_MAX_RMSE:
            return "warning"
        return "fail"

    @property
    def can_start(self):
        return self.quality_status in {"pass", "warning"}

    def quality_message(self):
        if self.quality_status == "pass":
            return f"Calibration passed (rmse={self.fit_rmse:.5f})."
        if self.quality_status == "warning":
            return f"Calibration warning accepted (rmse={self.fit_rmse:.5f})."
        if self.quality_status == "insufficient_points":
            return f"Calibration needs {self.required_points} display points; found {self.sample_count}."
        if self.quality_status == "insufficient_coverage":
            return "Calibration points must cover the center, corners, and edge centers."
        if self.quality_status == "fail":
            return f"Calibration failed (rmse={self.fit_rmse:.5f}, maximum={CALIBRATION_MAX_RMSE:.2f})."
        return "A valid calibration is required."

    def begin(self):
        self.is_collecting = True
        self.samples = []
        self.transform_matrix = None
        self.fit_rmse = None
        self.status_message = (
            f"Display calibration started. Look at target 1/{self.required_points}, then press C or Enter."
        )
        print(f"[Tobii Calibration] {self.status_message}")

    def cancel(self):
        self.is_collecting = False
        self.samples = []
        self.status_message = "Calibration canceled."
        print(f"[Tobii Calibration] {self.status_message}")

    def clear_saved(self):
        self.transform_matrix = None
        self.fit_rmse = None
        self.samples = []
        self.is_collecting = False
        for candidate_path in self._get_candidate_save_paths():
            if os.path.exists(candidate_path):
                os.remove(candidate_path)
        self.loaded_from_path = None
        self.status_message = "Saved calibration cleared."
        print(f"[Tobii Calibration] {self.status_message}")

    def undo(self):
        if self.samples:
            self.samples.pop()
            self.status_message = (
                f"Removed the last point. Look at target {self.sample_count + 1}/{self.required_points}."
            )
        else:
            self.status_message = "Nothing to undo."
        print(f"[Tobii Calibration] {self.status_message}")

    def capture_sample(self, raw_display_point, target_display_point):
        if not self.is_collecting or self.current_target_normalized is None:
            self.status_message = "Display calibration is not collecting a target. Press F6 to restart."
            print(f"[Tobii Calibration] {self.status_message}")
            return False
        if raw_display_point is None:
            self.status_message = "Not enough recent Tobii samples. Hold gaze still for a moment and try again."
            print(f"[Tobii Calibration] {self.status_message}")
            return False
        if target_display_point is None:
            self.status_message = "Window geometry is unavailable, so calibration capture cannot continue."
            print(f"[Tobii Calibration] {self.status_message}")
            return False

        self.samples.append(
            {
                "raw_display_point": [float(raw_display_point[0]), float(raw_display_point[1])],
                "target_display_point": [float(target_display_point[0]), float(target_display_point[1])],
                "window_normalized_point": list(self.current_target_normalized),
            }
        )

        if self.sample_count >= self.required_points:
            self._fit_transform()
            self.is_collecting = False
            if self.can_start:
                self.save()
                self.status_message = self.quality_message()
            else:
                self.status_message = self.quality_message() + " Press F6 to recalibrate."
        else:
            self.status_message = (
                f"Captured point {self.sample_count}/{self.required_points}. "
                f"Look at target {self.sample_count + 1}/{self.required_points}, then press C or Enter."
            )

        print(f"[Tobii Calibration] {self.status_message}")
        return True

    def _fit_transform(self):
        source_points = np.asarray(
            [sample["raw_display_point"] for sample in self.samples],
            dtype=np.float64,
        )
        target_points = np.asarray(
            [sample["target_display_point"] for sample in self.samples],
            dtype=np.float64,
        )
        design = np.column_stack([source_points, np.ones(source_points.shape[0], dtype=np.float64)])
        coeff_x, _, _, _ = np.linalg.lstsq(design, target_points[:, 0], rcond=None)
        coeff_y, _, _, _ = np.linalg.lstsq(design, target_points[:, 1], rcond=None)
        self.transform_matrix = np.vstack([coeff_x, coeff_y])
        predicted = design @ self.transform_matrix.T
        self.fit_rmse = float(np.sqrt(np.mean(np.sum((predicted - target_points) ** 2, axis=1))))

    def apply(self, raw_display_point):
        if raw_display_point is None:
            return None
        if self.transform_matrix is None:
            return float(raw_display_point[0]), float(raw_display_point[1])
        x, y = raw_display_point
        corrected = self.transform_matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
        return float(corrected[0]), float(corrected[1])

    def save(self):
        for candidate_path in self._get_candidate_save_paths():
            self._write_payload(candidate_path)
        self.loaded_from_path = self.calibration_path

    def load(self):
        last_error = None
        for candidate_path in self._get_candidate_load_paths():
            if not os.path.exists(candidate_path):
                continue
            try:
                with open(candidate_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("version") != GAZE_CALIBRATION_FORMAT_VERSION:
                    raise ValueError(
                        "Saved gaze calibration is incompatible with monitor-based coordinates. "
                        "Press F6 to create a new calibration."
                    )
                if payload.get("calibration_scope") != GAZE_CALIBRATION_SCOPE:
                    raise ValueError(f"Invalid gaze calibration scope in {candidate_path}")
                payload_participant_id = payload.get("participant_id")
                if self.participant_id and payload_participant_id != self.participant_id:
                    raise ValueError(
                        f"Calibration participant mismatch: expected {self.participant_id}, "
                        f"found {payload_participant_id!r}."
                    )
                matrix = np.asarray(payload.get("transform_matrix"), dtype=np.float64)
                if matrix.shape != (2, 3):
                    raise ValueError(f"Invalid gaze calibration matrix in {candidate_path}")
            except Exception as exc:
                last_error = exc
                if self._explicit_calibration_path is not None:
                    raise
                continue

            self.transform_matrix = matrix
            self.fit_rmse = payload.get("fit_rmse")
            self.samples = payload.get("samples", [])
            self.is_collecting = False
            self.loaded_from_path = candidate_path
            rmse_message = (
                f", rmse={self.fit_rmse:.5f}"
                if isinstance(self.fit_rmse, (float, int))
                else ""
            )
            quality_suffix = f" Quality={self.quality_status}."
            self.status_message = (
                f"Loaded participant display calibration from {candidate_path}{rmse_message}.{quality_suffix}"
            )
            print(f"[Tobii Calibration] {self.status_message}")
            return True

        if last_error is not None:
            raise last_error

        self.status_message = "No participant display calibration found. Press F6 to create one."
        return False


def get_surface_blit_rect(surface_size, window_size, max_scale=1.2):
    surface_width, surface_height = surface_size
    window_width, window_height = window_size
    scale = min(float(max_scale), window_width / surface_width, window_height / surface_height)
    scaled_width = max(1, int(round(surface_width * scale)))
    scaled_height = max(1, int(round(surface_height * scale)))
    offset_x = (window_width - scaled_width) // 2
    offset_y = (window_height - scaled_height) // 2
    return pygame.Rect(offset_x, offset_y, scaled_width, scaled_height)


def get_display_size():
    monitor_geometry = get_window_monitor_geometry()
    if monitor_geometry is not None:
        return monitor_geometry["width"], monitor_geometry["height"]
    display_info = pygame.display.Info()
    if display_info.current_w > 0 and display_info.current_h > 0:
        return int(display_info.current_w), int(display_info.current_h)
    if os.name == "nt":
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    return None


def get_window_monitor_geometry():
    if os.name != "nt":
        return None
    window_handle = pygame.display.get_wm_info().get("window")
    if not window_handle:
        return None

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("rcMonitor", ctypes.wintypes.RECT),
            ("rcWork", ctypes.wintypes.RECT),
            ("dwFlags", ctypes.wintypes.DWORD),
        ]

    user32 = ctypes.windll.user32
    user32.MonitorFromWindow.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.DWORD]
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    monitor_handle = user32.MonitorFromWindow(window_handle, 2)
    if not monitor_handle:
        return None

    monitor_info = MonitorInfo()
    monitor_info.cbSize = ctypes.sizeof(MonitorInfo)
    if not user32.GetMonitorInfoW(monitor_handle, ctypes.byref(monitor_info)):
        return None
    rect = monitor_info.rcMonitor
    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def get_window_client_geometry():
    if os.name != "nt":
        return None
    window_handle = pygame.display.get_wm_info().get("window")
    if not window_handle:
        return None

    rect = ctypes.wintypes.RECT()
    point = ctypes.wintypes.POINT()
    user32 = ctypes.windll.user32
    if not user32.GetClientRect(window_handle, ctypes.byref(rect)):
        return None
    if not user32.ClientToScreen(window_handle, ctypes.byref(point)):
        return None

    return {
        "left": int(point.x),
        "top": int(point.y),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def _get_display_geometry():
    monitor_geometry = get_window_monitor_geometry()
    if monitor_geometry is not None:
        return monitor_geometry
    display_size = get_display_size()
    if display_size is None:
        return None
    return {
        "left": 0,
        "top": 0,
        "width": int(display_size[0]),
        "height": int(display_size[1]),
    }


def surface_point_to_display_normalized(surface_point, surface_size, window_size):
    window_geometry = get_window_client_geometry()
    display_geometry = _get_display_geometry()
    if window_geometry is None or display_geometry is None:
        return None

    surface_rect = get_surface_blit_rect(surface_size, window_size)
    display_x = (
        window_geometry["left"]
        + surface_rect.left
        + (float(surface_point[0]) / float(surface_size[0])) * surface_rect.width
    )
    display_y = (
        window_geometry["top"]
        + surface_rect.top
        + (float(surface_point[1]) / float(surface_size[1])) * surface_rect.height
    )
    return (
        (display_x - display_geometry["left"]) / display_geometry["width"],
        (display_y - display_geometry["top"]) / display_geometry["height"],
    )


def window_point_to_display_normalized(window_point):
    window_geometry = get_window_client_geometry()
    display_geometry = _get_display_geometry()
    if window_geometry is None or display_geometry is None:
        return None
    display_x = window_geometry["left"] + float(window_point[0])
    display_y = window_geometry["top"] + float(window_point[1])
    return (
        (display_x - display_geometry["left"]) / display_geometry["width"],
        (display_y - display_geometry["top"]) / display_geometry["height"],
    )


def display_normalized_to_surface_point(display_point, surface_size, window_size):
    if display_point is None:
        return

    window_geometry = get_window_client_geometry()
    display_geometry = _get_display_geometry()
    if window_geometry is None or display_geometry is None:
        return None

    surface_rect = get_surface_blit_rect(surface_size, window_size)
    display_x = (
        display_geometry["left"] + float(display_point[0]) * display_geometry["width"]
    )
    display_y = (
        display_geometry["top"] + float(display_point[1]) * display_geometry["height"]
    )
    rel_x = (display_x - (window_geometry["left"] + surface_rect.left)) / surface_rect.width
    rel_y = (display_y - (window_geometry["top"] + surface_rect.top)) / surface_rect.height
    if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
        return None
    return rel_x * surface_size[0], rel_y * surface_size[1]


def window_to_surface_point(window_point, surface_size, window_size):
    surface_rect = get_surface_blit_rect(surface_size, window_size)
    if not surface_rect.collidepoint(window_point):
        return None

    rel_x = (window_point[0] - surface_rect.left) / surface_rect.width
    rel_y = (window_point[1] - surface_rect.top) / surface_rect.height
    return rel_x * surface_size[0], rel_y * surface_size[1]


def resolve_gaze_surface_point(surface_size, window_size, raw_gaze_point, gaze_calibration=None):
    if raw_gaze_point is None:
        return None, None

    corrected_display_point = (
        gaze_calibration.apply(raw_gaze_point)
        if gaze_calibration is not None
        else (float(raw_gaze_point[0]), float(raw_gaze_point[1]))
    )
    surface_point = display_normalized_to_surface_point(
        corrected_display_point,
        surface_size,
        window_size,
    )
    return surface_point, corrected_display_point


def evaluate_gaze_preview(
    gaze_tracker,
    gaze_calibration,
    window_size,
    *,
    minimum_samples=GAZE_PREVIEW_MIN_SAMPLES,
    window_sec=GAZE_PREVIEW_WINDOW_SEC,
):
    count_recent_samples = getattr(
        gaze_tracker,
        "get_recent_valid_gaze_sample_count",
        None,
    )
    if callable(count_recent_samples):
        sample_count = count_recent_samples(window_sec)
    else:
        cutoff_time = time.perf_counter() - float(window_sec)
        sample_count = sum(
            1
            for timestamp, _ in getattr(gaze_tracker, "_recent_gaze_points", ())
            if timestamp >= cutoff_time
        )
    raw_gaze_point = gaze_tracker.latest_gaze_point
    corrected_point = None
    if raw_gaze_point is not None:
        corrected_point = gaze_calibration.apply(raw_gaze_point)

    gaze_window_point = None
    if corrected_point is not None:
        window_geometry = get_window_client_geometry()
        display_geometry = _get_display_geometry() if window_geometry is not None else None
        if window_geometry is not None and display_geometry is not None:
            gaze_window_point = (
                display_geometry["left"]
                + corrected_point[0] * display_geometry["width"]
                - window_geometry["left"],
                display_geometry["top"]
                + corrected_point[1] * display_geometry["height"]
                - window_geometry["top"],
            )
        else:
            gaze_window_point = (
                corrected_point[0] * window_size[0],
                corrected_point[1] * window_size[1],
            )

    gaze_is_inside = bool(
        gaze_window_point is not None
        and 0 <= gaze_window_point[0] < window_size[0]
        and 0 <= gaze_window_point[1] < window_size[1]
    )
    valid = bool(sample_count >= int(minimum_samples) and gaze_is_inside)
    if raw_gaze_point is None:
        status = "gaze_not_detected"
    elif not gaze_is_inside:
        status = "gaze_outside_window"
    elif sample_count < int(minimum_samples):
        status = "collecting_samples"
    else:
        status = "ready"
    return {
        "valid": valid,
        "status": status,
        "sample_count": int(sample_count),
        "minimum_samples": int(minimum_samples),
        "window_sec": float(window_sec),
        "gaze_window_point": gaze_window_point,
    }


def draw_tobii_gaze_overlay(surface, surface_point):
    if surface_point is None:
        return

    gaze_x = int(round(surface_point[0]))
    gaze_y = int(round(surface_point[1]))

    if not (0 <= gaze_x < surface.get_width() and 0 <= gaze_y < surface.get_height()):
        return

    pygame.draw.circle(surface, (255, 255, 255), (gaze_x, gaze_y), 16, width=3)
    pygame.draw.circle(surface, (255, 60, 60), (gaze_x, gaze_y), 6)
    pygame.draw.line(surface, (255, 255, 255), (gaze_x - 10, gaze_y), (gaze_x + 10, gaze_y), width=2)
    pygame.draw.line(surface, (255, 255, 255), (gaze_x, gaze_y - 10), (gaze_x, gaze_y + 10), width=2)


def draw_gaze_calibration_overlay(
    surface,
    gaze_calibration,
    surface_rect=None,
    source_surface_size=None,
):
    if gaze_calibration is None:
        return

    surface.fill((0, 0, 0))
    title_font = pygame.font.Font(None, 52)
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)
    center_x = surface.get_width() // 2

    title = title_font.render("EYE TRACKER CALIBRATION", True, (245, 248, 252))
    surface.blit(title, title.get_rect(center=(center_x, 56)))

    if gaze_calibration.is_collecting:
        target_point = gaze_calibration.current_target_window_point(surface.get_size())
        for sample in gaze_calibration.samples:
            normalized_point = sample.get("window_normalized_point")
            if normalized_point is None:
                continue
            completed_point = (
                int(round(normalized_point[0] * surface.get_width())),
                int(round(normalized_point[1] * surface.get_height())),
            )
            pygame.draw.circle(surface, (70, 105, 125), completed_point, 8)

        if target_point is not None:
            point = (int(round(target_point[0])), int(round(target_point[1])))
            pygame.draw.circle(surface, (255, 255, 255), point, 23, width=3)
            pygame.draw.circle(surface, (255, 83, 83), point, 12)
            pygame.draw.circle(surface, (255, 255, 255), point, 3)

        progress = font.render(
            f"POINT {gaze_calibration.sample_count + 1} / {gaze_calibration.required_points}",
            True,
            (230, 235, 242),
        )
        surface.blit(progress, progress.get_rect(center=(center_x, surface.get_height() - 74)))
        instruction = small_font.render(
            "LOOK AT THE TARGET, THEN PRESS C OR ENTER    BACKSPACE: UNDO    ESC: CANCEL",
            True,
            (190, 200, 213),
        )
        surface.blit(instruction, instruction.get_rect(center=(center_x, surface.get_height() - 38)))
        return

    quality_status = gaze_calibration.quality_status
    is_warning = quality_status not in {"pass", "warning"}
    status_color = (255, 92, 92) if is_warning else (76, 211, 130)
    status_title = "CALIBRATION REQUIRED" if is_warning else "CALIBRATION READY"
    if "Drift check failed" in gaze_calibration.status_message:
        status_title = "GAZE CHECK WARNING"
        is_warning = True
        status_color = (255, 92, 92)

    status_surface = title_font.render(status_title, True, status_color)
    surface.blit(status_surface, status_surface.get_rect(center=(center_x, surface.get_height() // 2 - 55)))

    message = gaze_calibration.status_message or gaze_calibration.quality_message()
    message_lines = []
    current_line = []
    for word in message.split():
        candidate = " ".join(current_line + [word])
        if current_line and font.size(candidate)[0] > surface.get_width() - 100:
            message_lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        message_lines.append(" ".join(current_line))
    for line_index, line in enumerate(message_lines[:3]):
        message_surface = font.render(line, True, (235, 239, 245))
        surface.blit(
            message_surface,
            message_surface.get_rect(
                center=(center_x, surface.get_height() // 2 + 5 + line_index * 30)
            ),
        )

    controls = "F5: GAZE PREVIEW    F6: RECALIBRATE    F7: CLEAR SAVED CALIBRATION"
    controls_surface = small_font.render(controls, True, (190, 200, 213))
    surface.blit(
        controls_surface,
        controls_surface.get_rect(center=(center_x, surface.get_height() // 2 + 112)),
    )


def draw_gaze_drift_overlay(surface, elapsed_sec, duration_sec=DRIFT_CHECK_DURATION_SEC):
    center = (surface.get_width() // 2, surface.get_height() // 2)
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 185))
    surface.blit(veil, (0, 0))

    pulse = 1.0 - min(1.0, max(0.0, float(elapsed_sec) / float(duration_sec)))
    radius = int(round(18 + 8 * pulse))
    pygame.draw.circle(surface, (255, 255, 255), center, radius + 8, width=3)
    pygame.draw.circle(surface, (255, 80, 80), center, radius, width=4)
    pygame.draw.circle(surface, (255, 255, 255), center, 4)

    font = pygame.font.Font(None, 32)
    text = font.render("LOOK AT THE CENTER TARGET", True, (255, 255, 255))
    surface.blit(text, text.get_rect(center=(center[0], center[1] - 70)))


def _draw_gaze_preview_chef(surface, cell_rect):
    icon_size = max(28, int(min(cell_rect.width, cell_rect.height) * 0.5))
    center_x = cell_rect.centerx
    center_y = cell_rect.centery - 8
    head_radius = max(4, icon_size // 8)

    body_rect = pygame.Rect(0, 0, int(icon_size * 0.55), int(icon_size * 0.42))
    body_rect.midtop = (center_x, center_y)
    pygame.draw.rect(surface, (18, 174, 96), body_rect, border_radius=max(3, icon_size // 12))

    head_center = (center_x, center_y - head_radius)
    pygame.draw.circle(surface, (238, 188, 135), head_center, head_radius)
    hat_y = head_center[1] - head_radius
    hat_radius = max(4, icon_size // 9)
    for offset_x in (-hat_radius, 0, hat_radius):
        pygame.draw.circle(surface, (39, 211, 125), (center_x + offset_x, hat_y), hat_radius)
    hat_brim = pygame.Rect(0, 0, int(icon_size * 0.52), max(4, icon_size // 10))
    hat_brim.midbottom = (center_x, hat_y + hat_radius)
    pygame.draw.rect(surface, (25, 188, 106), hat_brim, border_radius=2)

    apron_rect = body_rect.inflate(-int(icon_size * 0.22), -int(icon_size * 0.08))
    apron_rect.midbottom = body_rect.midbottom
    pygame.draw.rect(surface, (245, 247, 242), apron_rect, border_radius=2)


def draw_gaze_preview_overlay(
    surface,
    preview_result,
):
    surface.fill((0, 0, 0))
    center = (surface.get_width() // 2, surface.get_height() // 2)
    preview_rect = pygame.Rect(
        70,
        90,
        surface.get_width() - 140,
        surface.get_height() - 210,
    )
    pygame.draw.rect(surface, (20, 24, 30), preview_rect, border_radius=8)
    pygame.draw.rect(surface, (100, 112, 128), preview_rect, width=2, border_radius=8)
    for column in range(1, 6):
        x = preview_rect.left + round(preview_rect.width * column / 6)
        pygame.draw.line(surface, (54, 62, 73), (x, preview_rect.top), (x, preview_rect.bottom), width=1)
    for row in range(1, 4):
        y = preview_rect.top + round(preview_rect.height * row / 4)
        pygame.draw.line(surface, (54, 62, 73), (preview_rect.left, y), (preview_rect.right, y), width=1)

    target_font = pygame.font.Font(None, 22)
    for label, column, row, relative_path in GAZE_PREVIEW_TARGETS:
        cell_left = preview_rect.left + round(preview_rect.width * column / 6)
        cell_right = preview_rect.left + round(preview_rect.width * (column + 1) / 6)
        cell_top = preview_rect.top + round(preview_rect.height * row / 4)
        cell_bottom = preview_rect.top + round(preview_rect.height * (row + 1) / 4)
        cell_rect = pygame.Rect(cell_left, cell_top, cell_right - cell_left, cell_bottom - cell_top)
        target_rect = cell_rect.inflate(-10, -10)
        pygame.draw.rect(surface, (35, 42, 51), target_rect, border_radius=7)

        if relative_path is None:
            _draw_gaze_preview_chef(surface, cell_rect)
        else:
            image_path = os.path.join(PROJECT_ROOT, relative_path)
            target_image = _GAZE_PREVIEW_IMAGE_CACHE.get(image_path)
            if target_image is None and os.path.exists(image_path):
                target_image = pygame.image.load(image_path)
                _GAZE_PREVIEW_IMAGE_CACHE[image_path] = target_image
            if target_image is not None:
                image_size = max(24, int(min(cell_rect.width, cell_rect.height) * 0.48))
                scaled_image = pygame.transform.scale(target_image, (image_size, image_size))
                image_rect = scaled_image.get_rect(
                    center=(cell_rect.centerx, cell_rect.centery - 8)
                )
                surface.blit(scaled_image, image_rect)

        label_surface = target_font.render(label, True, (235, 239, 245))
        surface.blit(
            label_surface,
            label_surface.get_rect(center=(cell_rect.centerx, cell_rect.bottom - 17)),
        )

    gaze_window_point = preview_result["gaze_window_point"]
    if gaze_window_point is not None and preview_result["status"] != "gaze_outside_window":
        draw_tobii_gaze_overlay(surface, gaze_window_point)

    title_font = pygame.font.Font(None, 42)
    prompt_font = pygame.font.Font(None, 27)
    title = title_font.render("LIVE GAZE PREVIEW", True, (245, 248, 252))
    status_text = "GAZE READY - PRESS F5 TO START"
    status_color = (76, 211, 130)
    if preview_result["status"] == "gaze_not_detected":
        status_text = "GAZE NOT DETECTED"
        status_color = (255, 92, 92)
    elif preview_result["status"] == "gaze_outside_window":
        status_text = "GAZE IS OUTSIDE THIS WINDOW"
        status_color = (255, 92, 92)
    elif preview_result["status"] == "collecting_samples":
        status_text = (
            f"COLLECTING GAZE SAMPLES  "
            f"{preview_result['sample_count']} / {preview_result['minimum_samples']}"
        )
        status_color = (245, 194, 66)
    status = prompt_font.render(status_text, True, status_color)
    controls = prompt_font.render(
        (
            "F5: START    F6: RECALIBRATE    F7: CLEAR SAVED CALIBRATION"
            if preview_result["valid"]
            else "F5: WAITING FOR VALID GAZE    F6: RECALIBRATE    F7: CLEAR SAVED CALIBRATION"
        ),
        True,
        (76, 211, 130) if preview_result["valid"] else (185, 194, 207),
    )
    surface.blit(title, title.get_rect(center=(center[0], 48)))
    surface.blit(status, status.get_rect(center=(center[0], surface.get_height() - 70)))
    surface.blit(controls, controls.get_rect(center=(center[0], surface.get_height() - 35)))
