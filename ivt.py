from dataclasses import dataclass
from bisect import bisect_left
import math
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class IVTConfig:
    median_window_samples: int = 3
    velocity_window_ms: float = 20.0
    velocity_threshold_deg_s: float = 30.0
    merge_gap_ms: float = 75.0
    merge_angle_deg: float = 0.5
    min_fixation_duration_ms: float = 60.0
    screen_diagonal_inches: float = 27.0
    screen_aspect_width: float = 16.0
    screen_aspect_height: float = 9.0
    viewing_distance_mm: float = 600.0
    foveal_threshold_deg: float = 2.0
    parafoveal_threshold_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.median_window_samples <= 0 or self.median_window_samples % 2 == 0:
            raise ValueError("median_window_samples must be a positive odd number")
        for field_name in (
            "velocity_window_ms",
            "velocity_threshold_deg_s",
            "merge_gap_ms",
            "merge_angle_deg",
            "min_fixation_duration_ms",
            "screen_diagonal_inches",
            "screen_aspect_width",
            "screen_aspect_height",
            "viewing_distance_mm",
            "foveal_threshold_deg",
            "parafoveal_threshold_deg",
        ):
            if float(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.parafoveal_threshold_deg <= self.foveal_threshold_deg:
            raise ValueError("parafoveal_threshold_deg must be greater than foveal_threshold_deg")

    @property
    def screen_size_mm(self) -> Tuple[float, float]:
        diagonal_mm = self.screen_diagonal_inches * 25.4
        aspect_diagonal = math.hypot(self.screen_aspect_width, self.screen_aspect_height)
        return (
            diagonal_mm * self.screen_aspect_width / aspect_diagonal,
            diagonal_mm * self.screen_aspect_height / aspect_diagonal,
        )

    @property
    def aspect_ratio_label(self) -> str:
        return f"{self.screen_aspect_width:g}:{self.screen_aspect_height:g}"


@dataclass
class IVTFixation:
    start_ms: int
    end_ms: int
    duration_ms: int
    centroid_x: float
    centroid_y: float
    sample_count: int
    mean_velocity_deg_s: float
    peak_velocity_deg_s: float
    samples: List[Dict[str, Any]]


def _gaze_vector(x: float, y: float, config: IVTConfig) -> Tuple[float, float, float]:
    screen_width_mm, screen_height_mm = config.screen_size_mm
    return (
        (float(x) - 0.5) * screen_width_mm,
        (0.5 - float(y)) * screen_height_mm,
        config.viewing_distance_mm,
    )


def angular_distance_deg(
    first: Tuple[float, float],
    second: Tuple[float, float],
    config: IVTConfig,
) -> float:
    first_vector = _gaze_vector(first[0], first[1], config)
    second_vector = _gaze_vector(second[0], second[1], config)
    dot_product = sum(a * b for a, b in zip(first_vector, second_vector))
    first_norm = math.sqrt(sum(component * component for component in first_vector))
    second_norm = math.sqrt(sum(component * component for component in second_vector))
    cosine = dot_product / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def visual_eccentricity_deg(
    gaze_surface_point: Tuple[float, float],
    target_surface_point: Tuple[float, float],
    surface_size: Tuple[int, int],
    map_display_size: Tuple[int, int],
    display_size: Tuple[int, int],
    config: Optional[IVTConfig] = None,
) -> float:
    config = config or IVTConfig()
    if min(*surface_size, *map_display_size, *display_size) <= 0:
        raise ValueError("surface, map display, and display dimensions must be positive")
    screen_width_mm, screen_height_mm = config.screen_size_mm
    delta_display_x = (
        (float(target_surface_point[0]) - float(gaze_surface_point[0]))
        * map_display_size[0]
        / surface_size[0]
    )
    delta_display_y = (
        (float(target_surface_point[1]) - float(gaze_surface_point[1]))
        * map_display_size[1]
        / surface_size[1]
    )
    delta_mm_x = delta_display_x * screen_width_mm / display_size[0]
    delta_mm_y = delta_display_y * screen_height_mm / display_size[1]
    distance_mm = math.hypot(delta_mm_x, delta_mm_y)
    return math.degrees(math.atan2(distance_mm, config.viewing_distance_mm))


def classify_vision_zone(
    eccentricity_deg: float,
    config: Optional[IVTConfig] = None,
) -> str:
    config = config or IVTConfig()
    if eccentricity_deg <= config.foveal_threshold_deg:
        return "foveal"
    if eccentricity_deg <= config.parafoveal_threshold_deg:
        return "parafoveal"
    return "peripheral"


def _valid_sample(sample: Dict[str, Any]) -> bool:
    if not bool(sample.get("valid", False)):
        return False
    try:
        return math.isfinite(float(sample["x"])) and math.isfinite(float(sample["y"]))
    except (KeyError, TypeError, ValueError):
        return False


def _split_valid_runs(samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    runs: List[List[Dict[str, Any]]] = []
    current_run: List[Dict[str, Any]] = []
    previous_timestamp: Optional[int] = None
    for source_sample in samples:
        sample = dict(source_sample)
        if not _valid_sample(sample):
            if current_run:
                runs.append(current_run)
                current_run = []
            previous_timestamp = None
            continue
        timestamp_ms = int(sample["timestamp_ms"])
        if previous_timestamp is not None and timestamp_ms <= previous_timestamp:
            if current_run:
                runs.append(current_run)
            current_run = []
        current_run.append(sample)
        previous_timestamp = timestamp_ms
    if current_run:
        runs.append(current_run)
    return runs


def _median_filter(run: List[Dict[str, Any]], window_size: int) -> None:
    radius = window_size // 2
    x_values = [float(sample["x"]) for sample in run]
    y_values = [float(sample["y"]) for sample in run]
    for index, sample in enumerate(run):
        start = max(0, index - radius)
        end = min(len(run), index + radius + 1)
        sample["filtered_x"] = float(median(x_values[start:end]))
        sample["filtered_y"] = float(median(y_values[start:end]))


def _closest_index(timestamps: Sequence[float], target_ms: float) -> int:
    insertion_index = bisect_left(timestamps, target_ms)
    if insertion_index <= 0:
        return 0
    if insertion_index >= len(timestamps):
        return len(timestamps) - 1
    before = insertion_index - 1
    if target_ms - timestamps[before] <= timestamps[insertion_index] - target_ms:
        return before
    return insertion_index


def _calculate_velocities(run: List[Dict[str, Any]], config: IVTConfig) -> None:
    half_window = config.velocity_window_ms / 2.0
    timestamps = [float(sample["timestamp_ms"]) for sample in run]
    for index, sample in enumerate(run):
        timestamp_ms = float(sample["timestamp_ms"])
        left_index = _closest_index(timestamps, timestamp_ms - half_window)
        right_index = _closest_index(timestamps, timestamp_ms + half_window)
        if left_index == right_index:
            if index == 0 and len(run) > 1:
                right_index = 1
            elif index > 0:
                left_index = index - 1
        elapsed_ms = float(run[right_index]["timestamp_ms"]) - float(run[left_index]["timestamp_ms"])
        if elapsed_ms <= 0:
            sample["velocity_deg_s"] = float("inf")
            continue
        angle = angular_distance_deg(
            (run[left_index]["filtered_x"], run[left_index]["filtered_y"]),
            (run[right_index]["filtered_x"], run[right_index]["filtered_y"]),
            config,
        )
        sample["velocity_deg_s"] = angle * 1000.0 / elapsed_ms


def _candidate_fixations(run: List[Dict[str, Any]], config: IVTConfig) -> List[List[Dict[str, Any]]]:
    candidates: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for sample in run:
        if float(sample["velocity_deg_s"]) < config.velocity_threshold_deg_s:
            current.append(sample)
        elif current:
            candidates.append(current)
            current = []
    if current:
        candidates.append(current)
    return candidates


def _centroid(samples: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        sum(float(sample["filtered_x"]) for sample in samples) / len(samples),
        sum(float(sample["filtered_y"]) for sample in samples) / len(samples),
    )


def _merge_candidates(
    candidates: Sequence[List[Dict[str, Any]]],
    config: IVTConfig,
) -> List[List[Dict[str, Any]]]:
    merged: List[List[Dict[str, Any]]] = []
    for candidate in candidates:
        if not merged:
            merged.append(list(candidate))
            continue
        previous = merged[-1]
        gap_ms = int(candidate[0]["timestamp_ms"]) - int(previous[-1]["timestamp_ms"])
        centroid_distance = angular_distance_deg(_centroid(previous), _centroid(candidate), config)
        if gap_ms <= config.merge_gap_ms and centroid_distance <= config.merge_angle_deg:
            previous.extend(candidate)
        else:
            merged.append(list(candidate))
    return merged


def classify_ivt_fixations(
    samples: Sequence[Dict[str, Any]],
    config: Optional[IVTConfig] = None,
) -> List[IVTFixation]:
    config = config or IVTConfig()
    all_candidates: List[List[Dict[str, Any]]] = []
    for run in _split_valid_runs(samples):
        _median_filter(run, config.median_window_samples)
        _calculate_velocities(run, config)
        all_candidates.extend(_candidate_fixations(run, config))

    fixations: List[IVTFixation] = []
    for candidate in _merge_candidates(all_candidates, config):
        start_ms = int(candidate[0]["timestamp_ms"])
        end_ms = int(candidate[-1]["timestamp_ms"])
        duration_ms = end_ms - start_ms
        if duration_ms < config.min_fixation_duration_ms:
            continue
        centroid_x, centroid_y = _centroid(candidate)
        velocities = [float(sample["velocity_deg_s"]) for sample in candidate]
        fixations.append(
            IVTFixation(
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                sample_count=len(candidate),
                mean_velocity_deg_s=sum(velocities) / len(velocities),
                peak_velocity_deg_s=max(velocities),
                samples=candidate,
            )
        )
    return fixations
