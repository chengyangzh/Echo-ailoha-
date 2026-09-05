from __future__ import annotations

import json
from pathlib import Path
from .evaluation import evaluate_retrieval


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    metrics = evaluate_retrieval(root / "eval" / "retrieval_controlled.json")
    print(json.dumps(metrics.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
