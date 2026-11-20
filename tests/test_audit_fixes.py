"""Regression tests for the publication audit."""
import pytest

from test_receptionist import BLANK, MONDAY_EVENING, RULES

import desk


@pytest.mark.parametrize("said,reading", [
    ("I don't want mornings", {}),
    ("I don’t want mornings", {"preferred_time": "mornings"}),
    ("I'd rather not come on Tuesday", {"preferred_day": "Tuesday"}),
])
def test_negated_wish_does_not_replace_a_preference(said, reading):
    fields = {"preferred_day": "Thursday", "preferred_time": "afternoon"}
    call = desk.Call("x", MONDAY_EVENING, fields=fields.copy())
    desk.turn(call, said, BLANK | reading, RULES)
    assert call.fields == fields


@pytest.mark.parametrize("reading", [{}, {"preferred_time": "I want mornings"}])
def test_positive_wish_after_a_negation_is_kept(reading):
    call = desk.Call("x", MONDAY_EVENING)
    desk.turn(call, "I don't want mornings, but I want Thursday afternoon.", BLANK | reading, RULES)
    assert desk.parse_time(call.fields["preferred_time"]) == (12, 17)
    assert desk.parse_day(call.fields["preferred_day"], MONDAY_EVENING).weekday() == 3
