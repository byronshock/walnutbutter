import pytest


@pytest.fixture(autouse=True)
def run_in_a_temporary_directory(tmp_path, monkeypatch):
    """Command-line runs checkpoint to runs/ in the working directory by default; keep that out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def forced_input(monkeypatch):
    """Pin forced drive: a test about the schedule, the clock or the plumbing wants a deterministic stimulus.

    INPUT_DRIVE is "rate" since September 14 (AUTHORITY.md §4.3), which makes the
    input a Poisson process, so every spike time and count becomes seed-dependent
    and assertions like "82 neurons fired" stop meaning anything. Tests that are
    about the input process itself set `drive` for themselves, and
    test_a_problem_and_the_command_line_choose_the_drive asserts the real default.
    """
    from walnutbutter import cli, constants, network
    for module in (constants, network, cli):
        monkeypatch.setattr(module, "INPUT_DRIVE", "forced", raising=False)
