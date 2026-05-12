"""External proof data loader (1Lab, agda-unimath, HoTT Coq, etc.)

For MVP, provides the loading interface contract. Future versions
will parse Agda/Rzk/Coq proof files and extract:
- Theorem statements and proofs
- Lemma dependency graphs
- Tactic sequences
- Higher-categorical structures (paths, equivalences, coherences)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class DataLoader:
    """Loads proof data from external sources for training/evaluation."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None

    def load_jsonl(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Load proof records from JSONL file."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_problems(self, filepath: str) -> List[Dict[str, Any]]:
        """Load proof problems from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    def save_trajectories(
        self, trajectories: List[Any], filepath: str,
    ):
        """Save proof trajectories to JSONL format."""
        with open(filepath, "w", encoding="utf-8") as f:
            for traj in trajectories:
                records = traj.to_records() if hasattr(traj, "to_records") else [traj]
                for rec in records:
                    rec_clean = {}
                    for k, v in rec.items():
                        if hasattr(v, "__dict__"):
                            rec_clean[k] = {sk: str(sv) for sk, sv in v.__dict__.items()}
                        else:
                            rec_clean[k] = str(v) if not isinstance(v, (int, float, bool)) else v
                    f.write(json.dumps(rec_clean) + "\n")

    def load_agda_module(self, filepath: str) -> Dict[str, Any]:
        """Stub: parse Agda module structure.

        Future: extract data/record types, function signatures, proofs,
        cubical primitives (hcomp, transp, Glue), and path structures.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agda module not found: {filepath}")
        return {
            "file": filepath,
            "status": "parsed (stub)",
            "definitions": [],
            "theorems": [],
        }
