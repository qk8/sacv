"""
tests/unit/test_lesson_learned_schema.py
=========================================
Ensures LessonLearned TypedDict has all required fields.
"""
from __future__ import annotations

import pytest
from sacv.orchestration.state import LessonLearned


def test_lesson_learned_required_fields():
    """Ensure LessonLearned can be created with all required fields."""
    lesson = LessonLearned(
        task_id="t",
        pattern_discovered="x",
        negative_constraints=[],
        blast_radius_learned={},
        correction_type="none",
        session_duration_ms=0,
    )
    # Access every field — will raise KeyError if any is missing
    assert lesson["task_id"] == "t"
    assert lesson["pattern_discovered"] == "x"
    assert lesson["negative_constraints"] == []
    assert lesson["blast_radius_learned"] == {}
    assert lesson["correction_type"] == "none"
    assert lesson["session_duration_ms"] == 0


def test_lesson_learned_all_fields_present():
    """Verify no required field is missing from LessonLearned."""
    lesson = LessonLearned(
        task_id="task-1",
        pattern_discovered="module=backend-domain | resolved_in=2_attempts",
        negative_constraints=["[SECURITY] injection risk"],
        blast_radius_learned={"entry_files": ["A.java"], "affected_files": ["B.java"]},
        correction_type="self_correction",
        session_duration_ms=12345,
    )
    expected_keys = {
        "task_id", "pattern_discovered", "negative_constraints",
        "blast_radius_learned", "correction_type", "session_duration_ms",
    }
    assert set(lesson.keys()) == expected_keys, (
        f"LessonLearned missing keys: {expected_keys - set(lesson.keys())}"
    )
