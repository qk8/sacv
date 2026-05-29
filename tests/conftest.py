"""
pytest configuration: marks and shared fixtures.
"""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests; no I/O")
    config.addinivalue_line("markers", "integration: requires stub providers; no live APIs")
    config.addinivalue_line("markers", "e2e: full graph via VCR cassettes; no live APIs")
