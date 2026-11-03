"""RNG-neutral, line-oriented progress reporting for gaze behavior jobs."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any, Mapping, TextIO


def process_memory() -> dict[str, float]:
    """Return Linux process memory counters without adding a dependency."""
    values: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmSize:", "VmHWM:")):
                    key, raw = line.split(":", 1)
                    values[key] = float(raw.split()[0]) / 1024.0
    except OSError:
        pass
    return {
        "rss_mib": values.get("VmRSS", float("nan")),
        "vms_mib": values.get("VmSize", float("nan")),
        "peak_rss_mib": values.get("VmHWM", float("nan")),
    }


def _memory_text() -> str:
    memory = process_memory()
    return (
        f"rss_mib={memory['rss_mib']:.1f} "
        f"vms_mib={memory['vms_mib']:.1f} "
        f"peak_rss_mib={memory['peak_rss_mib']:.1f}"
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    value = max(int(float(seconds)), 0)
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


@dataclass
class BootstrapTracker:
    reporter: "ProgressReporter"
    group: str
    total: int
    started: float
    completed: int = 0
    next_report: int = 0

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 0)
        self.next_report = min(
            self.reporter.bootstrap_progress_every, self.total
        )

    def update(self, completed: int) -> None:
        self.completed = min(max(int(completed), self.completed), self.total)
        while self.next_report and self.completed >= self.next_report:
            self.reporter.bootstrap_update(
                self.group, self.next_report, self.total, self.started
            )
            if self.next_report == self.total:
                self.next_report = 0
            else:
                self.next_report = min(
                    self.next_report + self.reporter.bootstrap_progress_every,
                    self.total,
                )

    def done(self) -> None:
        self.update(self.total)
        self.reporter.emit(
            f"[DONE][{self.group}] bootstrap={self.total} "
            f"elapsed={format_duration(self.reporter.clock() - self.started)}"
        )


class ProgressReporter:
    """Formatting-only observer; deliberately imports no model or RNG package."""

    def __init__(
        self, *, enabled: bool, total_rollout_episodes: int,
        progress_every_episodes: int = 1,
        bootstrap_progress_every: int = 2000,
        stream: TextIO | None = None,
        clock=time.monotonic,
    ) -> None:
        if int(progress_every_episodes) <= 0:
            raise ValueError("progress_every_episodes must be positive")
        if int(bootstrap_progress_every) <= 0:
            raise ValueError("bootstrap_progress_every must be positive")
        self.enabled = bool(enabled)
        self.total_rollout_episodes = max(int(total_rollout_episodes), 0)
        self.progress_every_episodes = int(progress_every_episodes)
        self.bootstrap_progress_every = int(bootstrap_progress_every)
        self.stream = sys.stdout if stream is None else stream
        self.clock = clock
        self.started = self.clock()
        self.rollout_started = self.started
        self.completed_rollout_episodes = 0
        self.completed_models = 0
        self.completed_analyses = 0
        self.context: dict[str, Any] = {
            "phase": "preflight", "model": None,
            "condition": None, "episode": None,
        }

    def emit(self, message: str = "") -> None:
        if self.enabled:
            print(message, file=self.stream, flush=True)

    def banner(self, *, model_count: int, condition_count: int, evaluation, output) -> None:
        self.emit("=" * 60)
        self.emit("GAZE BEHAVIOR SUITE")
        self.emit("=" * 60)
        self.emit(f"models             : {model_count}")
        self.emit(f"conditions/model   : {condition_count}")
        self.emit(f"episodes/condition : {evaluation.episodes}")
        self.emit(f"rollout episodes   : {self.total_rollout_episodes}")
        self.emit(f"decision times     : {','.join(map(str, evaluation.decision_times))}")
        self.emit(f"horizons           : {','.join(map(str, evaluation.horizons))}")
        self.emit(f"episode length     : {evaluation.episode_length}")
        self.emit(f"gaze mode          : {evaluation.gaze_mode}")
        self.emit(f"action mode        : {evaluation.action_mode}")
        self.emit(f"bootstrap          : {evaluation.bootstrap_iterations}")
        self.emit(f"device             : {evaluation.device}")
        self.emit(f"output             : {output}")
        self.emit("=" * 60)

    def phase(self, number: int, name: str) -> None:
        self.context.update(
            {"phase": name.lower().replace(" ", "_"), "condition": None,
             "episode": None}
        )
        if int(number) == 2:
            self.rollout_started = self.clock()
        self.emit(f"[PHASE {number}/4] {name}")

    def preflight_model(self, index: int, total: int, name: str, loaded) -> None:
        self.context["model"] = name
        self.emit(f"[MODEL {index}/{total}] {name}")
        self.emit(f"  checkpoint: {loaded.path}")
        self.emit(f"  checkpoint_sha256: {loaded.checkpoint_sha256}")
        self.emit(f"  actor_sha256: {loaded.actor_sha256}")
        self.emit(f"  gaze_sha256: {loaded.gaze_sha256}")
        self.emit(
            f"  memory_lifetime_ms: {loaded.metadata['memory_lifetime_ms']}"
        )
        self.emit("[PASS] checkpoint contracts")
        self.emit("[PASS] fixed partner")

    def start_model(self, index: int, total: int, name: str) -> float:
        self.context.update({"model": name, "condition": None, "episode": None})
        self.emit(f"[MODEL {index}/{total}] {name}")
        return self.clock()

    def start_condition(
        self, *, index: int, total: int, decision_time: int, horizon: int,
        episodes: int, episode_length: int,
    ) -> float:
        self.context.update(
            {"condition": f"T{decision_time}/H{horizon}", "episode": None}
        )
        self.emit(f"[COND {index}/{total}] T{decision_time} H{horizon}")
        self.emit(
            f"[START] episodes={episodes}, episode_length={episode_length}"
        )
        return self.clock()

    def episode(
        self, *, model: str, decision_time: int, horizon: int,
        episode: int, episodes: int, deliveries_mean: float,
        acquisitions_per_decision: float, last_episode: Mapping[str, Any],
    ) -> None:
        self.completed_rollout_episodes += 1
        self.context["episode"] = f"{episode}/{episodes}"
        if episode % self.progress_every_episodes and episode != episodes:
            return
        elapsed = self.clock() - self.rollout_started
        eta = None
        if self.completed_rollout_episodes:
            remaining = max(
                self.total_rollout_episodes - self.completed_rollout_episodes, 0
            )
            eta = elapsed / self.completed_rollout_episodes * remaining
        condition_percent = 100.0 * episode / max(episodes, 1)
        global_percent = 100.0 * self.completed_rollout_episodes / max(
            self.total_rollout_episodes, 1
        )
        self.emit(
            f"[ROLLOUT] model={model} T={decision_time} H={horizon} "
            f"episode={episode}/{episodes} condition={condition_percent:.1f}% "
            f"global={self.completed_rollout_episodes}/"
            f"{self.total_rollout_episodes} ({global_percent:.1f}%) "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)} "
            f"deliveries_mean={deliveries_mean:.3f} "
            f"acquisitions/decision={acquisitions_per_decision:.3f}"
        )
        self.emit(
            "  last_episode: "
            f"deliveries={last_episode['deliveries']} "
            f"sparse_reward={last_episode['sparse_team_reward']} "
            f"acquisitions={last_episode['acquisitions']} "
            f"requests={last_episode['gaze_requests']}"
        )

    def done_condition(
        self, *, decision_time: int, horizon: int, episodes: int,
        started: float, deliveries: float, acquisitions_per_decision: float,
        forced_stay: float, next_condition: str,
    ) -> None:
        self.emit(
            f"[DONE] T{decision_time} H{horizon} episodes={episodes} "
            f"elapsed={format_duration(self.clock() - started)} "
            f"deliveries={deliveries:.3f} "
            f"acq/decision={acquisitions_per_decision:.3f} "
            f"forced_stay={forced_stay:.3f} next={next_condition}"
        )

    def model_done(self, name: str, episodes: int, started: float) -> None:
        self.completed_models += 1
        self.emit(
            f"[MODEL DONE] {name} rollout episodes={episodes} "
            f"elapsed={format_duration(self.clock() - started)}"
        )

    def reuse(self, model: str, path, *, episodes: int, acquisitions: int) -> None:
        self.context.update({"phase": "rollout_reuse", "model": model})
        self.completed_rollout_episodes += int(episodes)
        self.emit(
            f"[ROLLOUT][REUSE] model={model} loading existing dataset: {path}"
        )
        self.emit(f"  episodes={episodes} acquisitions={acquisitions}")

    def account_existing(self, episodes: int) -> None:
        self.completed_rollout_episodes += int(episodes)

    def skip_rollout(self, reason: str) -> None:
        self.emit(f"[PHASE 2/4][ROLLOUT][SKIP] {reason}")

    def skip_analysis(self, reason: str) -> None:
        self.emit(f"[PHASE 3/4][ANALYSIS][SKIP] {reason}")

    def start_analysis(
        self, *, model: str, name: str, index: int, total: int,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> float:
        self.context.update(
            {"phase": f"analysis:{name}", "model": model,
             "condition": None, "episode": None}
        )
        self.emit(
            f"[PHASE 3/4][ANALYSIS][{index}/{total}] {name} "
            f"model={model}"
        )
        self.emit("[START]")
        self.emit(f"  {_memory_text()}")
        if diagnostics:
            self.emit(
                "  " + " ".join(
                    f"{key}={value}" for key, value in diagnostics.items()
                )
            )
        return self.clock()

    def done_analysis(
        self, name: str, started: float,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.completed_analyses += 1
        self.emit(
            f"[DONE] {name} elapsed={format_duration(self.clock() - started)} "
            f"{_memory_text()}"
        )
        if diagnostics:
            self.emit(
                "  " + " ".join(
                    f"{key}={value}" for key, value in diagnostics.items()
                )
            )

    def skip_analysis_stage(self, name: str, reason: str) -> None:
        self.completed_analyses += 1
        self.emit(f"[SKIP] {name} reason={reason!r} {_memory_text()}")

    def bootstrap_tracker(self, group: str, total: int) -> BootstrapTracker:
        return BootstrapTracker(self, group, int(total), self.clock())

    def bootstrap_update(
        self, group: str, completed: int, total: int, started: float
    ) -> None:
        elapsed = self.clock() - started
        eta = None if completed <= 0 else elapsed / completed * (total - completed)
        percent = 100.0 * completed / max(total, 1)
        self.emit(
            f"[BOOTSTRAP][{group}] {completed}/{total} ({percent:.1f}%) "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)} "
            f"{_memory_text()}"
        )

    def ablation_plan(
        self, *, requests: int, variants: int, total_queries: int
    ) -> float:
        self.emit(
            f"[ANALYSIS][input_ablation] requests={requests} "
            f"variants={variants} total_queries={total_queries}"
        )
        return self.clock()

    def ablation_update(
        self, completed: int, total: int, started: float
    ) -> None:
        elapsed = self.clock() - started
        eta = None if completed <= 0 else elapsed / completed * (total - completed)
        self.emit(
            f"[ABLATION] {completed}/{total} "
            f"({100.0 * completed / max(total, 1):.1f}%) "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}"
        )

    def cross_step(self, index: int, total: int, name: str) -> None:
        self.context.update({"phase": f"cross_model:{name}"})
        self.emit(f"[{index}/{total}] {name}")

    def cross_done(self) -> None:
        self.emit("[DONE] cross-model report")

    def failure(self, error: BaseException) -> None:
        self.emit("[FAIL]")
        self.emit(f"phase     : {self.context.get('phase')}")
        self.emit(f"model     : {self.context.get('model')}")
        self.emit(f"condition : {self.context.get('condition')}")
        self.emit(f"episode   : {self.context.get('episode')}")
        self.emit(f"elapsed   : {format_duration(self.clock() - self.started)}")
        self.emit(f"error     : {error}")

    def finish(
        self, *, output, summary, bootstrap: int,
        models_completed: int, model_count: int,
    ) -> None:
        self.emit("=" * 60)
        self.emit("GAZE BEHAVIOR SUITE COMPLETE")
        self.emit("=" * 60)
        self.emit(
            f"models completed : {models_completed}/{model_count}"
        )
        self.emit(
            f"rollout episodes : {self.completed_rollout_episodes}/"
            f"{self.total_rollout_episodes}"
        )
        self.emit(f"analyses         : {self.completed_analyses}")
        self.emit(f"bootstrap        : {bootstrap}")
        self.emit(f"elapsed          : {format_duration(self.clock() - self.started)}")
        self.emit(f"output           : {output}")
        self.emit(f"summary          : {summary}")
        self.emit("=" * 60)
