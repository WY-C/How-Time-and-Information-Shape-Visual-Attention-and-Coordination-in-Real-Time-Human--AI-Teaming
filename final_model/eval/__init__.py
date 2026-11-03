"""Read-only Full Action policy evaluation and behavioral diagnostics."""

from __future__ import annotations

__all__ = ["EvaluationGazeController"]


def __getattr__(name: str):
    if name != "EvaluationGazeController":
        raise AttributeError(name)

    import torch

    from final_model.partial_ver.training import LearnedGazeController
    from final_model.server_parallel.gaze_v1 import action_index_to_tile

    class EvaluationGazeController(LearnedGazeController):
        """Compatibility export for the frozen legacy ``final_model.eval_sweep``."""

        def __call__(self, memory, fixation, context):
            observation = self.spatial_input(memory, fixation)
            with torch.no_grad():
                distribution = self.actor.distribution(
                    torch.as_tensor(
                        observation[None], dtype=torch.float32, device=self.device
                    ),
                    torch.as_tensor(
                        context[None], dtype=torch.float32, device=self.device
                    ),
                )
                index = int(distribution.probs.argmax(-1).item())
            return action_index_to_tile(index)

        def reset(self):
            self.active = None
            self.completed = []

    globals()[name] = EvaluationGazeController
    return EvaluationGazeController
