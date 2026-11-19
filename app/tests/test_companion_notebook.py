"""The companion notebook recomputes the app's numbers independently; every check must match.

Runs the notebook's code cells (skipping the Colab setup cell) in one namespace, about 10 seconds.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "ACMAD_OND_IDR_App_Companion.ipynb"


def test_companion_notebook_matches_the_app(monkeypatch):
    nb = json.loads(NOTEBOOK.read_text())
    code = [
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "setup" not in c.get("metadata", {}).get("tags", [])
    ]
    assert not any(line.lstrip().startswith("!") for src in code for line in src.splitlines()), \
        "shell commands belong in the setup cell only"
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "path", [str(REPO), *sys.path])
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    ns: dict = {"__name__": "__companion__"}
    for src in code:
        exec(compile(src, str(NOTEBOOK), "exec"), ns)
    checks = ns["CHECKS"]
    failed = [c for c in checks if not c[-1]]
    assert len(checks) >= 100
    assert not failed, failed
