import json
from experiments.programme_closure import evaluate as E

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(path, value):
    write(path, value)
    path.with_suffix(".sha256").write_text(E.sha(path) + "\n")
