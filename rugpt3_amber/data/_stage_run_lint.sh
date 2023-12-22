#!/bin/bash

set -x

echo -e '\n'
echo 'Running lint check...'

source venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"
FAILED=0

lint_output=$(python -m pylint --exit-zero --rcfile rugpt3_amber/data/.pylintrc rugpt3_amber/generator.py)

python rugpt3_amber/data/lint_level.py \
  --lint-output "${lint_output}" \
  --target-score "10"

if [[ $? -ne 0 ]]; then
  echo "Lint check failed."
  FAILED=1
else
  echo "Lint check passed."
fi
