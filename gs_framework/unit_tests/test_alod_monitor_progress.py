from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "alod" / "monitor_capped_training.py"
SPEC = importlib.util.spec_from_file_location("monitor_capped_training", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_parses_official_tqdm_progress_with_carriage_return() -> None:
    text = (
        "\rTraining progress:  90%|█████████ | 27070/30000 "
        "[29:36<03:00, 16.21it/s, Loss=0.1060818]"
    )
    iteration, nodes, progressed = MONITOR.parse_log_progress(text, None, None)
    assert (iteration, nodes, progressed) == (27070, None, True)


def test_repeated_tqdm_iteration_is_not_new_progress() -> None:
    text = "\rTraining progress:  90%|█████████ | 27070/30000 [29:37<03:00]"
    iteration, nodes, progressed = MONITOR.parse_log_progress(text, 27070, None)
    assert (iteration, nodes, progressed) == (27070, None, False)


def test_preserves_legacy_iteration_and_creator_size_parsing() -> None:
    text = "[ITER 12] loss=0.2\nSize=256/1024\n"
    iteration, nodes, progressed = MONITOR.parse_log_progress(text, None, None)
    assert (iteration, nodes, progressed) == (12, 256, True)
