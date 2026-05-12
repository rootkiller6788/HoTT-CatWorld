"""Smoke tests for CLI."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import tempfile


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "hottworld.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "simulate" in result.stdout.lower() or "search" in result.stdout.lower()


def test_cli_info():
    result = subprocess.run(
        [sys.executable, "-m", "hottworld.cli", "info"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_cli_simulate():
    result = subprocess.run(
        [sys.executable, "-m", "hottworld.cli", "simulate",
         "--problem", "simple_equality", "--max-steps", "5"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "simple_equality" in result.stdout or "Proof" in result.stdout


def test_cli_search():
    result = subprocess.run(
        [sys.executable, "-m", "hottworld.cli", "search",
         "--problem", "path_composition", "--simulations", "10", "--max-depth", "5"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_cli_generate():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = os.path.join(tmpdir, "test_trajs.jsonl")
        result = subprocess.run(
            [sys.executable, "-m", "hottworld.cli", "generate",
             "--samples", "10", "--output", output],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(output)


def test_cli_train():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "hottworld.cli", "train",
             "--samples", "20", "--output", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))
