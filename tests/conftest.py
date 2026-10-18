import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def record(request):
    """Write measured Gate-0 quantities to $RVM_GATE0_OUT/<test_file>__<test_name>.json."""
    out = os.environ.get("RVM_GATE0_OUT")

    def _rec(**values):
        if not out:
            return
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        stem = Path(request.node.fspath).stem
        name = request.node.originalname or request.node.name
        f = p / f"{stem}__{name}.json"
        prev = json.loads(f.read_text()) if f.exists() else {}
        prev.update(values)
        f.write_text(json.dumps(prev, indent=2, default=float))

    return _rec


@pytest.fixture(scope="session")
def cfg():
    from rvm_lora.config import load_config

    return load_config()
