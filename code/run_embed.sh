#!/bin/bash
cd "$(dirname "$0")/.."
until grep -q DONE logs/embed.log 2>/dev/null; do caffeinate -i .venv/bin/python code/embed_corpus.py >> logs/embed.log 2>&1; sleep 5; done
