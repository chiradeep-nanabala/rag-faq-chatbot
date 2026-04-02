#!/bin/bash
# One-time setup: create virtual environment and install dependencies

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Setup complete. To start the chatbot:"
echo "  source .venv/bin/activate"
echo "  export OPENAI_API_KEY=sk-..."
echo "  python chatbot.py"
