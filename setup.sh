#!/bin/bash
# One-time setup: install dependencies using uv

uv sync
echo ""
echo "Setup complete. To start the chatbot:"
echo "  export OPENAI_API_KEY=sk-..."
echo "  uv run python chatbot.py"
