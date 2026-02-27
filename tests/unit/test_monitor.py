"""Unit tests for QRE live monitor."""

import sqlite3
import time
import os
from pathlib import Path

import pytest


class TestFindActiveRuns:
    """Test discovery of active optimizer runs."""

    def test_finds_recent_db(self, tmp_path):
        """DB modified recently → detected as active run."""
        from qre.monitor import find_active_runs

        run_dir = tmp_path / "2026-02-21_04-08-52_calmar-btc-v3"
        cp_dir = run_dir / "checkpoints"
        cp_dir.mkdir(parents=True)
        db_path = cp_dir / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=100, best_value=5.0)

        runs = find_active_runs(tmp_path, max_age_seconds=300)
        assert len(runs) == 1
        assert runs[0]["run_name"] == "2026-02-21_04-08-52_calmar-btc-v3"
        assert len(runs[0]["db_files"]) == 1

    def test_ignores_old_db(self, tmp_path):
        """DB not modified for > max_age_seconds → not detected."""
        from qre.monitor import find_active_runs

        run_dir = tmp_path / "2026-02-20_old-run"
        cp_dir = run_dir / "checkpoints"
        cp_dir.mkdir(parents=True)
        db_path = cp_dir / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=10, best_value=1.0)
        old_time = time.time() - 600
        os.utime(db_path, (old_time, old_time))

        runs = find_active_runs(tmp_path, max_age_seconds=300)
        assert len(runs) == 0

    def test_multiple_dbs_in_one_run(self, tmp_path):
        """Multiple .db files in one run → grouped together."""
        from qre.monitor import find_active_runs

        run_dir = tmp_path / "2026-02-21_multi-symbol"
        cp_dir = run_dir / "checkpoints"
        cp_dir.mkdir(parents=True)
        _create_mock_optuna_db(cp_dir / "optuna_BTC.db", n_trials=50, best_value=3.0)
        _create_mock_optuna_db(cp_dir / "optuna_SOL.db", n_trials=30, best_value=2.0)

        runs = find_active_runs(tmp_path, max_age_seconds=300)
        assert len(runs) == 1
        assert len(runs[0]["db_files"]) == 2

    def test_no_results_dir(self, tmp_path):
        """Non-existent results dir → empty list, no error."""
        from qre.monitor import find_active_runs

        runs = find_active_runs(tmp_path / "nonexistent", max_age_seconds=300)
        assert runs == []

    def test_filter_by_name(self, tmp_path):
        """Partial name filter → only matching runs."""
        from qre.monitor import find_active_runs

        for name in ["2026-02-21_btc-run", "2026-02-21_sol-run"]:
            cp_dir = tmp_path / name / "checkpoints"
            cp_dir.mkdir(parents=True)
            _create_mock_optuna_db(cp_dir / "optuna_X.db", n_trials=10, best_value=1.0)

        runs = find_active_runs(tmp_path, max_age_seconds=300, name_filter="btc")
        assert len(runs) == 1
        assert "btc" in runs[0]["run_name"]


class TestQueryDbStats:
    """Test querying Optuna DB for trial statistics."""

    def test_basic_stats(self, tmp_path):
        """Query returns correct trial counts and best value."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=100, best_value=5.0)

        stats = query_db_stats(db_path)
        assert stats.symbol == "BTC"
        assert stats.completed > 0
        assert stats.pruned > 0
        assert stats.best_value == pytest.approx(5.0)
        assert stats.n_trials_requested == 1000

    def test_best_params_present(self, tmp_path):
        """Best trial params are populated."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=100, best_value=5.0)

        stats = query_db_stats(db_path)
        assert "macd_fast" in stats.best_params
        assert "rsi_period" in stats.best_params

    def test_user_attrs_present(self, tmp_path):
        """User attrs from best trial are populated."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=100, best_value=5.0)

        stats = query_db_stats(db_path)
        assert "sharpe_equity" in stats.user_attrs
        assert "max_drawdown" in stats.user_attrs

    def test_trials_per_min(self, tmp_path):
        """Trials per minute is calculated."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_BTC.db"
        _create_mock_optuna_db(db_path, n_trials=100, best_value=5.0)

        stats = query_db_stats(db_path)
        assert stats.trials_per_min is None or stats.trials_per_min >= 0

    def test_corrupted_db(self, tmp_path):
        """Corrupted DB → returns None, no crash."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_BTC.db"
        db_path.write_text("not a database")

        stats = query_db_stats(db_path)
        assert stats is None

    def test_symbol_from_filename(self, tmp_path):
        """Symbol is parsed from DB filename."""
        from qre.monitor import query_db_stats

        db_path = tmp_path / "optuna_SOL.db"
        _create_mock_optuna_db(db_path, n_trials=50, best_value=3.0)

        stats = query_db_stats(db_path)
        assert stats.symbol == "SOL"


class TestFormatParams:
    """Test param formatting for display."""

    def test_returns_dict_with_three_keys(self):
        """format_params returns dict with macd, rsi, trend keys."""
        from qre.monitor import format_params

        params = {
            "macd_fast": 2.5, "macd_slow": 25.0, "macd_signal": 5.0,
            "rsi_period": 14.0, "rsi_lower": 30.0, "rsi_upper": 70.0,
            "rsi_lookback": 4.0, "trend_tf": 0,
        }
        result = format_params(params)
        assert isinstance(result, dict)
        assert "macd" in result
        assert "rsi" in result
        assert "trend" in result

    def test_macd_fast_float_formatting(self):
        """macd_fast float is formatted to 1 decimal."""
        from qre.monitor import format_params

        params = {
            "macd_fast": 2.5, "macd_slow": 25.0, "macd_signal": 5.0,
            "rsi_period": 14.0, "rsi_lower": 30.0, "rsi_upper": 70.0,
            "rsi_lookback": 4.0, "trend_tf": 0,
        }
        result = format_params(params)
        assert "2.5" in result["macd"]

    def test_trend_tf_maps_index_to_label(self):
        """trend_tf index 0→4h, 1→8h, 2→1d."""
        from qre.monitor import format_params

        params = {
            "macd_fast": 2.5, "macd_slow": 25.0, "macd_signal": 5.0,
            "rsi_period": 14.0, "rsi_lower": 30.0, "rsi_upper": 70.0,
            "rsi_lookback": 4.0, "trend_tf": 2,
        }
        result = format_params(params)
        assert result["trend"] == "1d"

    def test_missing_params_use_question_mark(self):
        """Missing params show '?' placeholder."""
        from qre.monitor import format_params

        result = format_params({})
        assert "?" in result["macd"]


class TestRenderSymbolPanel:
    """Test Rich Panel rendering for a symbol."""

    def test_renders_without_error(self):
        """Panel renders successfully with full stats."""
        from qre.monitor import render_symbol_panel, SymbolStats

        stats = SymbolStats(
            symbol="BTC",
            completed=1250,
            pruned=100,
            failed=5,
            n_trials_requested=5000,
            best_value=1.2345,
            best_trial_number=847,
            best_params={
                "macd_fast": 2.5, "macd_slow": 25.0, "macd_signal": 5.0,
                "rsi_period": 14.0, "rsi_lower": 30.0, "rsi_upper": 70.0,
                "rsi_lookback": 4.0, "trend_tf": 0,
            },
            user_attrs={
                "sharpe_equity": 1.85, "max_drawdown": 12.3,
                "total_pnl_pct": 142.5, "trades": 186, "trades_per_year": 93.0,
            },
            trials_per_min=4.2,
            eta_minutes=15.0,
        )
        panel = render_symbol_panel(stats)
        assert panel is not None

    def test_renders_with_new_best(self):
        """Panel shows NEW marker when best improves."""
        from rich.console import Console
        from io import StringIO
        from qre.monitor import render_symbol_panel, SymbolStats

        stats = SymbolStats(
            symbol="BTC", completed=100, best_value=2.0, best_trial_number=50,
            best_params={"macd_fast": 2.5}, user_attrs={"sharpe_equity": 1.5},
        )
        console = Console(file=StringIO(), force_terminal=True)
        panel = render_symbol_panel(stats, prev_best=1.0)
        console.print(panel)
        output = console.file.getvalue()
        assert "NEW" in output

    def test_renders_without_metrics(self):
        """Panel renders gracefully when user_attrs is empty (legacy run)."""
        from qre.monitor import render_symbol_panel, SymbolStats

        stats = SymbolStats(
            symbol="SOL", completed=50, best_value=1.0, best_trial_number=10,
            best_params={"macd_fast": 3.0},
        )
        panel = render_symbol_panel(stats)
        assert panel is not None

    def test_renders_with_warm_start(self):
        """Panel shows warm start source when present."""
        from rich.console import Console
        from io import StringIO
        from qre.monitor import render_symbol_panel, SymbolStats

        stats = SymbolStats(
            symbol="BTC", completed=100, best_value=2.0, best_trial_number=50,
            best_params={"macd_fast": 2.5}, warm_start_source="calmar-btc-v2",
        )
        console = Console(file=StringIO(), force_terminal=True)
        panel = render_symbol_panel(stats)
        console.print(panel)
        output = console.file.getvalue()
        assert "calmar-btc-v2" in output


def _create_mock_optuna_db(path: Path, n_trials: int, best_value: float):
    """Create a minimal Optuna-compatible SQLite DB for testing."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.execute("CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT)")
    cur.execute("INSERT INTO studies VALUES (1, 'test_study')")

    cur.execute("CREATE TABLE study_user_attributes (study_user_attribute_id INTEGER PRIMARY KEY, study_id INTEGER, key TEXT, value_json TEXT)")
    cur.execute("INSERT INTO study_user_attributes VALUES (1, 1, 'n_trials_requested', '1000')")
    cur.execute("INSERT INTO study_user_attributes VALUES (2, 1, 'symbol', '\"BTC/USDC\"')")

    cur.execute("""CREATE TABLE trials (
        trial_id INTEGER PRIMARY KEY, number INTEGER, study_id INTEGER,
        state TEXT, datetime_start TEXT, datetime_complete TEXT
    )""")

    cur.execute("""CREATE TABLE trial_values (
        trial_value_id INTEGER PRIMARY KEY, trial_id INTEGER,
        objective INTEGER, value REAL, value_type TEXT
    )""")

    cur.execute("""CREATE TABLE trial_params (
        trial_param_id INTEGER PRIMARY KEY, trial_id INTEGER,
        param_name TEXT, param_value REAL, distribution_json TEXT
    )""")

    cur.execute("""CREATE TABLE trial_user_attributes (
        trial_user_attribute_id INTEGER PRIMARY KEY, trial_id INTEGER,
        key TEXT, value_json TEXT
    )""")

    for i in range(n_trials):
        state = "COMPLETE" if i % 5 != 0 else "PRUNED"
        cur.execute(
            "INSERT INTO trials VALUES (?, ?, 1, ?, '2026-02-21 04:00:00', '2026-02-21 04:01:00')",
            (i + 1, i, state),
        )
        if state == "COMPLETE":
            value = best_value if i == n_trials - 1 else best_value * 0.5
            cur.execute(
                "INSERT INTO trial_values VALUES (?, ?, 0, ?, 'FLOAT')",
                (i + 1, i + 1, value),
            )

    best_trial_id = n_trials
    for name, val in [("macd_fast", 2.5), ("macd_slow", 16), ("macd_signal", 6),
                      ("rsi_period", 5), ("rsi_lower", 34), ("rsi_upper", 63),
                      ("rsi_lookback", 5), ("trend_tf", 2), ("trend_strict", 1), ("allow_flip", 1)]:
        cur.execute(
            "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) VALUES (?, ?, ?, '{}')",
            (best_trial_id, name, val),
        )

    for key, val in [("sharpe_equity", "2.49"), ("max_drawdown", "-7.56"),
                     ("total_pnl_pct", "79.4"), ("trades", "291"), ("trades_per_year", "144.41")]:
        cur.execute(
            "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
            (best_trial_id, key, val),
        )

    conn.commit()
    conn.close()
