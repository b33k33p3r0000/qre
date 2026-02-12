"""QRE Discord Agent — reports optimization status via /run command.

Lightweight agent that listens on Discord and responds to /run
with current QRE optimization progress.

Usage:
    python discord_agent.py

Requires DISCORD_BOT_TOKEN and DISCORD_GUILD_ID in qre/.env
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

QRE_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = QRE_ROOT / "results"
LOGS_DIR = QRE_ROOT / "logs"

logger = logging.getLogger("qre.discord_agent")


def _find_running_qre() -> dict | None:
    """Check if a QRE process is running, return info dict or None."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "python.*-m qre"],
            capture_output=True, text=True, timeout=5,
        )
        if not result.stdout.strip():
            return None

        line = result.stdout.strip().split("\n")[0]
        pid = int(line.split()[0])

        # Parse coin from command line
        coin = "BTC"  # default
        parts = line.lower()
        if "--sol" in parts or "sol" in parts:
            coin = "SOL"
        if "--btc" in parts:
            coin = "BTC"

        # Get process start time
        stat_result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True, text=True, timeout=5,
        )
        elapsed = stat_result.stdout.strip() if stat_result.returncode == 0 else "?"

        return {"pid": pid, "coin": coin, "elapsed": elapsed, "cmdline": line}
    except Exception as e:
        logger.warning(f"Process check failed: {e}")
        return None


def _find_latest_study(coin: str) -> dict | None:
    """Find the latest Optuna study DB and read progress."""
    try:
        import optuna

        if not RESULTS_DIR.exists():
            return None

        # Find most recent run directory for this coin
        run_dirs = sorted(
            [d for d in RESULTS_DIR.iterdir()
             if d.is_dir() and coin in d.name],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            return None

        latest = run_dirs[0]
        db_path = latest / "checkpoints" / f"optuna_{coin}.db"
        if not db_path.exists():
            return None

        storage_url = f"sqlite:///{db_path}"
        storage = optuna.storages.RDBStorage(url=storage_url)
        studies = optuna.study.get_all_study_summaries(storage=storage)
        if not studies:
            return None

        summary = studies[0]
        n_complete = summary.n_trials
        best_value = summary.best_trial.value if summary.best_trial else None

        # Try to get total trials from log
        n_total = _parse_total_trials(latest.name)

        return {
            "run_dir": latest.name,
            "n_complete": n_complete,
            "n_total": n_total,
            "best_value": best_value,
        }
    except Exception as e:
        logger.warning(f"Study read failed: {e}")
        return None


def _parse_total_trials(run_dir_name: str) -> int | None:
    """Try to find total trial count from the log file."""
    try:
        # Logs are named qre_YYYY-MM-DD_HH-MM-SS.log
        # Run dirs are named YYYY-MM-DD_HH-MM-SS_COIN
        # Match by timestamp prefix
        ts_prefix = run_dir_name.rsplit("_", 1)[0]  # e.g. 2026-02-12_15-30-45

        log_files = sorted(LOGS_DIR.glob("qre_*.log"), reverse=True)
        for log_file in log_files:
            if ts_prefix in log_file.name:
                # Read last 100 lines looking for trial count
                result = subprocess.run(
                    ["tail", "-100", str(log_file)],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.split("\n"):
                    if "trials" in line.lower() and "optimization" in line.lower():
                        # "Starting AWF optimization: 10000 trials"
                        for word in line.split():
                            if word.isdigit() and int(word) > 100:
                                return int(word)
                break
    except Exception:
        pass
    return None


def _guess_preset(n_total: int | None) -> str:
    """Guess preset name from trial count."""
    if n_total is None:
        return "Unknown"
    presets = {
        2000: "Quick (~15m)",
        5000: "Standard (~45m)",
        10000: "Production (~90m)",
        15000: "Deep (~3h)",
        25000: "Uber (~6h)",
    }
    return presets.get(n_total, f"Custom ({n_total} trials)")


def main():
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    guild_id = os.getenv("DISCORD_GUILD_ID", "")

    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not set in .env")
        sys.exit(1)
    if not guild_id:
        print("ERROR: DISCORD_GUILD_ID not set in .env")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)
    tree = app_commands.CommandTree(bot)
    guild = discord.Object(id=int(guild_id))

    @tree.command(
        name="run",
        description="Check QRE optimization status",
        guild=guild,
    )
    async def run_status(interaction: discord.Interaction):
        proc = _find_running_qre()

        if proc is None:
            # Not running — show last completed run
            for coin in ["BTC", "SOL"]:
                study = _find_latest_study(coin)
                if study:
                    lines = [
                        "\U0001f52c **QRE Optimization**",
                        "\u251c\u2500\u2500 Status: \u23f9\ufe0f Not running",
                        f"\u251c\u2500\u2500 Last run: {study['run_dir']}",
                        f"\u251c\u2500\u2500 Trials: {study['n_complete']}",
                    ]
                    if study["best_value"] is not None:
                        lines.append(
                            f"\u2514\u2500\u2500 Best value: {study['best_value']:.4f}"
                        )
                    await interaction.response.send_message("\n".join(lines))
                    return

            await interaction.response.send_message(
                "\U0001f52c **QRE Optimization**\n"
                "\u2514\u2500\u2500 Status: \u23f9\ufe0f Not running (no previous runs found)"
            )
            return

        # Running — show live progress
        coin = proc["coin"]
        study = _find_latest_study(coin)

        lines = [
            "\U0001f52c **QRE Optimization**",
            f"\u251c\u2500\u2500 Status: \u2705 Running (PID {proc['pid']})",
            f"\u251c\u2500\u2500 Coin: {coin}",
        ]

        if study:
            preset = _guess_preset(study["n_total"])
            lines.append(f"\u251c\u2500\u2500 Preset: {preset}")

            if study["n_total"]:
                pct = (study["n_complete"] / study["n_total"]) * 100
                lines.append(
                    f"\u251c\u2500\u2500 Progress: {study['n_complete']} / "
                    f"{study['n_total']} trials ({pct:.0f}%)"
                )
            else:
                lines.append(
                    f"\u251c\u2500\u2500 Progress: {study['n_complete']} trials"
                )

            if study["best_value"] is not None:
                lines.append(
                    f"\u251c\u2500\u2500 Best value: {study['best_value']:.4f} (Sharpe)"
                )

        lines.append(f"\u2514\u2500\u2500 Elapsed: {proc['elapsed']}")
        await interaction.response.send_message("\n".join(lines))

    @bot.event
    async def on_ready():
        await tree.sync(guild=guild)
        logger.info(f"QRE Agent online as {bot.user}")

    logger.info("Starting QRE Discord Agent...")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
