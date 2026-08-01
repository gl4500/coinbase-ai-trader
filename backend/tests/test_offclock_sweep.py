from tools import offclock_sweep as sweep


def test_config_grid_is_16_configs():
    grid = sweep._config_grid()
    assert len(grid) == 16
    # 2 substrates x 2 label variants x 4 horizons, all distinct
    assert len(set(grid)) == 16
    assert ("dollar", "direction", 4) in grid
    assert ("time", "triple_barrier", 168) in grid


def test_render_results_doc_has_row_per_config():
    rows = [
        {
            "substrate": "dollar",
            "label_variant": "direction",
            "horizon": 4,
            "auc": 0.55,
            "n": 5000,
            "precision": True,
            "expected_return": True,
            "paper_sharpe": False,
            "ece": True,
            "recommended_tau": 0.6,
        },
        {
            "substrate": "time",
            "label_variant": "direction",
            "horizon": 4,
            "auc": 0.51,
            "n": 5000,
            "precision": False,
            "expected_return": False,
            "paper_sharpe": False,
            "ece": True,
            "recommended_tau": float("nan"),
        },
    ]
    doc = sweep._render_results_doc(rows)
    assert "# Off-the-Clock Sweep Results" in doc
    assert doc.count("| dollar ") >= 1
    assert doc.count("| time ") >= 1
    # the dollar-minus-time delta section pairs the matched configs
    assert "Dollar - time delta" in doc
    assert "| direction | 4 |" in doc


def test_render_results_doc_empty_rows():
    doc = sweep._render_results_doc([])
    assert "# Off-the-Clock Sweep Results" in doc
    assert "no configs" in doc.lower()
