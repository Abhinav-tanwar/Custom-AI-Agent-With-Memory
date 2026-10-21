#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Create the Python environment and install requirements first. See README.md.'
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py
