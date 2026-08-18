from __future__ import annotations

import pytest


@pytest.mark.determinism("d1")
def test_governed_top_level_cleaning_config_is_rejected(deterministic_seed: int, tmp_path) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.orchestrator import run_dataprep

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("timestamp,close\n2026-01-01,1.0\n", encoding="utf-8")

    with pytest.raises(Exception, match="pipeline.cleaning"):
        run_dataprep(
            {
                "cleaning": {
                    "governance_mode": "governed",
                    "combos": [
                        {
                            "name": "default",
                            "steps": [],
                        }
                    ],
                },
                "data": {"input_path": str(csv_path)},
                "pipeline": {
                    "spec_inline": {
                        "ops": [],
                    }
                },
            },
            backtest_metric=None,
        )
