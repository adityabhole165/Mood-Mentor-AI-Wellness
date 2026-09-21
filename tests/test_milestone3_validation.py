from pathlib import Path

import src.milestone3_validation as validation


def test_validation_main_returns_zero_when_pytest_passes(monkeypatch):
    class Result:
        returncode = 0

    monkeypatch.setattr(validation.subprocess, "run", lambda *args, **kwargs: Result())

    assert validation.main() == 0


def test_validation_main_returns_nonzero_when_pytest_fails(monkeypatch):
    class Result:
        returncode = 3

    monkeypatch.setattr(validation.subprocess, "run", lambda *args, **kwargs: Result())

    assert validation.main() == 3
