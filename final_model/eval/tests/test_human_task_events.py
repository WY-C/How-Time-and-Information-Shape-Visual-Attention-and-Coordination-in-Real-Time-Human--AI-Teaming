import json

from final_model.eval.human_task_events import final_onion_transition_is_ai_owned


def _pots(two=(2, 2), ready=()):
    onion = {"2_items": [list(two)]} if two else {}
    if ready:
        onion["ready"] = [list(ready)]
    return json.dumps({"onion": onion, "empty": []})


def test_simultaneous_human_interact_cannot_be_attributed_to_ai_without_ai_interact():
    pre = _pots(two=(2, 2))
    post = _pots(two=None, ready=(2, 2))
    assert not final_onion_transition_is_ai_owned(
        ai_held="onion", ai_action="stay", ai_facing_tile=(2, 2),
        pre_pot_states=pre, post_pot_states=post, next_ai_held=None,
    )


def test_ai_deposit_requires_ai_facing_target_pot_transition():
    pre = _pots(two=(2, 2))
    post = _pots(two=None, ready=(2, 2))
    assert final_onion_transition_is_ai_owned(
        ai_held="onion", ai_action="interact", ai_facing_tile=(2, 2),
        pre_pot_states=pre, post_pot_states=post, next_ai_held=None,
    )
    assert not final_onion_transition_is_ai_owned(
        ai_held="onion", ai_action="interact", ai_facing_tile=(1, 1),
        pre_pot_states=pre, post_pot_states=post, next_ai_held=None,
    )
