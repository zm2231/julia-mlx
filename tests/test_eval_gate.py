import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.environ.get("JULIA_CHECKPOINT"), reason="set JULIA_CHECKPOINT to run the typed-decisions gate")
def test_typed_decisions_reproduces_published_result():
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(ROOT / "evals"))
    from common import ENGINE, agreement
    from typed_decisions import REFERENCE, gate_failures, load_questions, score

    from julia_mlx import load_model

    questions = load_questions()
    logits = load_model(os.environ["JULIA_CHECKPOINT"], **ENGINE).logits([question["row"] for question in questions])
    report = {**score(questions, logits), **agreement([question["id"] for question in questions], logits, json.loads(REFERENCE.read_text()))}
    assert gate_failures(report) == []


def test_gate_rejects_any_single_disagreement():
    sys.path.insert(0, str(ROOT / "evals"))
    from typed_decisions import PUBLISHED, gate_failures, score

    reference = {"q1": [2.0, 1.0], "q2": [0.0, 3.0]}
    questions = [{"id": "q1", "row": {"type": "choice"}, "keys": ["a", "b"], "gold": "a"},
                 {"id": "q2", "row": {"type": "noul"}, "keys": ["false", "true"], "gold": "true"}]
    from common import agreement

    ids = ["q1", "q2"]
    passing = {"by_type": dict(PUBLISHED), "total": 2, **agreement(ids, [[2.0, 1.0], [0.0, 3.0]], reference)}
    assert gate_failures(passing) == []
    flipped = {**passing, **agreement(ids, [[1.0, 2.0], [0.0, 3.0]], reference)}
    assert any("winner differs" in failure for failure in gate_failures(flipped))
    drifted = {**passing, **agreement(ids, [[2.0, 1.0], [0.0, 3.01]], reference)}
    assert any("logit delta" in failure for failure in gate_failures(drifted))
    assert any("per-type" in failure for failure in gate_failures({**passing, "by_type": {**PUBLISHED, "noul": 482}}))
    assert score(questions, [[2.0, 1.0], [0.0, 3.0]])["correct"] == 2
    with pytest.raises(ValueError, match="has 1 logits"):
        agreement(ids, [[2.0, 1.0], [0.0, 3.0]], {"q1": [2.0], "q2": [0.0, 3.0]})
    with pytest.raises(ValueError, match="cover every"):
        agreement(ids, [[2.0, 1.0], [0.0, 3.0]], {"q1": [2.0, 1.0]})


def test_cli_refuses_to_overwrite_its_comparator(tmp_path):
    import subprocess

    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    hard_link = tmp_path / "alias.json"
    os.link(reference, hard_link)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(reference)
    for script in ("typed_decisions.py", "pilots.py"):
        for write in (str(reference), str(tmp_path / "." / "reference.json"), str(hard_link), str(symlink)):
            process = subprocess.run([sys.executable, str(ROOT / "evals" / script), "unused-checkpoint", "--reference", str(reference), "--write-reference", write],
                                     capture_output=True, text=True, check=False)
            assert process.returncode != 0 and "must differ" in process.stderr
    assert reference.read_text() == "{}"


@pytest.mark.skipif(not os.environ.get("JULIA_CHECKPOINT"), reason="set JULIA_CHECKPOINT to run the classification pilots")
def test_pilots_reproduce_published_result():
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(ROOT / "evals"))
    from common import ENGINE
    from pilots import REFERENCE, evaluate, gate_failures

    from julia_mlx import load_model

    report, _ = evaluate(load_model(os.environ["JULIA_CHECKPOINT"], **ENGINE), json.loads(REFERENCE.read_text()))
    assert gate_failures(report) == []
