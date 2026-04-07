#!/bin/bash
export PYTHONIOENCODING=utf-8
export ANTHROPIC_API_KEY=$(cat "$(dirname "$0")/.anthropic_api_key")
cd /Users/ritaholloway/Desktop/self-correction-2026

python3 main.py --dataset humaneval-x --language java --np 5 --max_attempts 10 --model_name claude-sonnet-4-5-20250929 --mode iterative --refine_mode critique+refine > logs/claude_java_critique_refine.log 2>&1
echo "critique+refine done"

python3 main.py --dataset humaneval-x --language java --np 5 --max_attempts 10 --model_name claude-sonnet-4-5-20250929 --mode iterative --refine_mode critique+history+refine > logs/claude_java_critique_history_refine.log 2>&1
echo "critique+history+refine done"
