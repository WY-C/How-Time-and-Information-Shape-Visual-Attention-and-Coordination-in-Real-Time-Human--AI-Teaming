import copy
import os
from dataclasses import dataclass
from typing import Optional


TUTORIAL_TIME_CONSTRAINTS_MS = (200, 400, 600)
TUTORIAL_PATH_STEPS = (0, 2, 4)
TUTORIAL_TEST_DURATION_MS = 80000
TUTORIAL_TEST_SCORE_TARGET = 100
TUTORIAL_DEFAULT_TIME_CONSTRAINT_MS = 400
TUTORIAL_DEFAULT_PATH_STEPS = 2
TUTORIAL_TEST_DECISION_STEPS = (
    TUTORIAL_TEST_DURATION_MS // TUTORIAL_DEFAULT_TIME_CONSTRAINT_MS
)
TUTORIAL_PATH_MODEL_FILENAMES = {
    "cramped_room": "tutorial_beta_cramped_room.pt",
    "bothway_coordination": "bothway_coordination_1000000.pt",
    "coordination_ring": "coordination_ring_1000000.pt",
}


@dataclass(frozen=True)
class TutorialStage:
    number: int
    layout: str
    ai_agent: str
    checkpoint_layout: Optional[str]
    checkpoint_subpath: Optional[str]
    requires_test: bool
    uses_path_model: bool


TUTORIAL_STAGES = (
    TutorialStage(1, "cramped_room", "GREEDY", None, None, False, True),
    TutorialStage(
        2,
        "bothway_coordination",
        "HSP",
        "random0_medium",
        os.path.join("hsp", "s2", "hsp-S2-s36", "3.pt"),
        True,
        True,
    ),
    TutorialStage(
        3,
        "coordination_ring",
        "HSP",
        "random1",
        os.path.join("hsp", "s2", "hsp-S2-s36", "2.pt"),
        True,
        True,
    ),
)


def configure_tutorial_trial(
    args,
    stage,
    phase,
    project_root,
    checkpoint_resolver,
):
    is_test = phase == "test"
    use_path_model = stage.uses_path_model
    args.condition = ""
    args.layout = stage.layout
    args.time_constraint = TUTORIAL_DEFAULT_TIME_CONSTRAINT_MS
    args.path_step = TUTORIAL_DEFAULT_PATH_STEPS
    args.visualization_mode = "full"
    args.path_render_style = "highlight"
    args.heatmap_source = "model" if use_path_model else "rollout"
    args.enable_model_prediction = use_path_model
    args.path_model_path = (
        os.path.join(
            project_root,
            "predict_path",
            "output",
            TUTORIAL_PATH_MODEL_FILENAMES[stage.layout],
        )
        if use_path_model
        else ""
    )
    args.ai_agent = stage.ai_agent
    args._tutorial_ai_checkpoint = checkpoint_resolver(stage)
    args._tutorial_ai_algorithm = None
    args._tutorial_phase = phase
    args._tutorial_test_duration_ms = TUTORIAL_TEST_DURATION_MS
    args._tutorial_test_score_target = TUTORIAL_TEST_SCORE_TARGET
    args._status_condition = f"TUTORIAL STAGE {stage.number}"
    args.fixed_duration_ms = TUTORIAL_TEST_DURATION_MS if is_test else None
    if is_test:
        args.decision_horizon = TUTORIAL_TEST_DECISION_STEPS
    args._wait_for_block_start = True
    args._start_on_any_key = True
    args._start_wait_prompt = (
        "PRESS ANY KEY TO START TEST"
        if is_test
        else "PRESS ANY KEY TO START TUTORIAL"
    )
    args.enable_experiment_logging = False


def run_tutorial(args, run_trial, configure_trial):
    stage_index = 0
    practice_time_constraint = TUTORIAL_DEFAULT_TIME_CONSTRAINT_MS
    practice_path_steps = TUTORIAL_DEFAULT_PATH_STEPS

    while stage_index < len(TUTORIAL_STAGES):
        stage = TUTORIAL_STAGES[stage_index]
        practice_args = copy.deepcopy(args)
        configure_trial(practice_args, stage, "practice")
        practice_args.time_constraint = practice_time_constraint
        practice_args.path_step = practice_path_steps
        practice_args.visualization_mode = "none" if practice_path_steps == 0 else "full"
        print(
            f"[Tutorial] stage={stage.number} layout={stage.layout} "
            "phase=practice | F1=layout F2=time F3=path Q=continue"
        )
        transition = run_trial(practice_args)
        if transition is None:
            return
        practice_time_constraint = int(
            transition.get("time_constraint", practice_time_constraint)
        )
        practice_path_steps = int(transition.get("path_steps", practice_path_steps))

        if transition.get("type") == "layout":
            stage_index = (stage_index + 1) % len(TUTORIAL_STAGES)
            continue

        if not stage.requires_test:
            stage_index += 1
            continue

        while True:
            test_args = copy.deepcopy(args)
            configure_trial(test_args, stage, "test")
            print(
                f"[Tutorial] stage={stage.number} layout={stage.layout} "
                f"phase=test duration={TUTORIAL_TEST_DURATION_MS // 1000}s "
                f"target={TUTORIAL_TEST_SCORE_TARGET}"
            )
            result = run_trial(test_args)
            if result is None:
                return
            if result.get("passed"):
                break

        stage_index += 1

    print("[Tutorial] completed")
