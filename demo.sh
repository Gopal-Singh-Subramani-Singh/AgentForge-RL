#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================"
echo "AI SOFTWARE TEAM DEMO"
echo "========================================"

echo
echo "[1/3] Pre-flight check"
python tests/demo_smoke_test.py

echo
echo "[2/3] Cleaning workspace"
rm -rf workspaces/rl_bank_demo

echo
echo "[3/3] Running RL-scheduled LLM team"
echo

PYTHONUNBUFFERED=1 \
python -m integration.rl_bank_runner \
    | tee results/demo_bank_latest.log
