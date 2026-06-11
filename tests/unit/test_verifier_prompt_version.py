"""
tests/unit/test_verifier_prompt_version.py
============================================
CFG-003: Verify verifier classifier system prompt has version prefix.
"""
from __future__ import annotations


class TestVerifierClassifierPromptVersion:

    def test_classifier_prompt_has_version_constant(self):
        """_CLASSIFIER_SYSTEM_PROMPT is prefixed with # prompt_version: <version>."""
        from sacv.nodes.verifier import _CLASSIFIER_SYSTEM_PROMPT, _CLASSIFIER_SYSTEM_VERSION
        expected_prefix = f"# prompt_version: {_CLASSIFIER_SYSTEM_VERSION}\n"
        assert _CLASSIFIER_SYSTEM_PROMPT.startswith(expected_prefix), (
            f"Expected prompt to start with '{expected_prefix!r}', "
            f"got: {_CLASSIFIER_SYSTEM_PROMPT[:80]!r}"
        )

    def test_version_constant_is_non_empty_string(self):
        """_CLASSIFIER_SYSTEM_VERSION is a non-empty string."""
        from sacv.nodes.verifier import _CLASSIFIER_SYSTEM_VERSION
        assert isinstance(_CLASSIFIER_SYSTEM_VERSION, str)
        assert len(_CLASSIFIER_SYSTEM_VERSION) > 0

    def test_classifier_prompt_is_non_empty(self):
        """_CLASSIFIER_SYSTEM_PROMPT contains meaningful content."""
        from sacv.nodes.verifier import _CLASSIFIER_SYSTEM_PROMPT
        result = _CLASSIFIER_SYSTEM_PROMPT
        assert len(result) > 50
        assert "classifier" in result.lower()
