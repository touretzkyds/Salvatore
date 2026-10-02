#!/bin/zsh

TOOL_DIR="${0:A:h}"
PYTHON_BIN="/Users/yuyuanzhengjack/Desktop/Touretzky_Dev/Salvatore/.venv/bin/python"

cd "$TOOL_DIR" || exit 1
exec "$PYTHON_BIN" annotate.py --images-dir images --output-dir output
