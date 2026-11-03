"""Full mode uses the one shared ActionNet implementation."""

from final_model.common.mappo import ACTION_ARCHITECTURE_ID, build_actionnet

__all__ = ["ACTION_ARCHITECTURE_ID", "build_actionnet"]

