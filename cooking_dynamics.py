"""Experiment-specific cooking dynamics."""

from typing import Any


DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP = 5
_COOK_RATE_MARKER = "_onion_cook_ticks_per_env_step"


def configure_onion_cook_rate(
    mdp: Any,
    ticks_per_env_step: int = DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP,
) -> Any:
    """Make cooking soups containing onion advance by the requested tick rate.

    The base Overcooked dynamics already advance a cooking soup by one tick per
    environment transition. This wrapper adds only the remaining ticks, keeping
    the original recipe cook-time threshold unchanged.
    """
    ticks_per_env_step = int(ticks_per_env_step)
    if ticks_per_env_step < 1:
        raise ValueError("ticks_per_env_step must be at least 1")

    configured_rate = getattr(mdp, _COOK_RATE_MARKER, None)
    if configured_rate is not None:
        if configured_rate != ticks_per_env_step:
            raise ValueError(
                "Onion cook rate is already configured as "
                f"{configured_rate} ticks per environment step"
            )
        return mdp

    original_step_environment_effects = mdp.step_environment_effects

    def step_environment_effects(state: Any) -> Any:
        result = original_step_environment_effects(state)

        extra_ticks = ticks_per_env_step - 1
        if extra_ticks <= 0:
            return result

        for obj in state.objects.values():
            if getattr(obj, "name", None) != "soup":
                continue

            # Current SoupState representation.
            if hasattr(obj, "cook") and hasattr(obj, "ingredients"):
                if (
                    not getattr(obj, "is_cooking", False)
                    or "onion" not in obj.ingredients
                ):
                    continue
                for _ in range(extra_ticks):
                    if not obj.is_cooking:
                        break
                    obj.cook()
                continue

            # Legacy ObjectState representation used by this experiment.
            soup_state = getattr(obj, "state", None)
            if not isinstance(soup_state, tuple) or len(soup_state) != 3:
                continue
            soup_type, num_items, cooking_tick = soup_state
            if (
                soup_type != "onion"
                or num_items != mdp.num_items_for_soup
                or cooking_tick >= mdp.soup_cooking_time
            ):
                continue
            x, y = obj.position
            if mdp.terrain_mtx[y][x] != "P":
                continue
            obj.state = (
                soup_type,
                num_items,
                min(cooking_tick + extra_ticks, mdp.soup_cooking_time),
            )

        return result

    mdp.step_environment_effects = step_environment_effects
    setattr(mdp, _COOK_RATE_MARKER, ticks_per_env_step)
    return mdp
