"""
FAQ RAG Chatbot — main entry point.

Usage:
    python chatbot.py [--faq faq.md] [--model gpt-4o-mini]
"""

import os
import argparse
from openai import OpenAI
from rag import FAQRetriever

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """\
You are a helpful customer support assistant. Answer the user's question using \
ONLY the FAQ context provided below. Be concise and friendly.

If the context does not contain a relevant answer, say:
"I'm sorry, I don't have information about that. Please contact our support team."

Do NOT make up information or answer from general knowledge.

FAQ Context:
{context}
"""


def build_context(results: list[dict]) -> str:
    """Format retrieved FAQ entries into a context string."""
    parts = []
    for r in results:
        parts.append(f"Q: {r['question']}\nA: {r['answer']}")
    return "\n\n".join(parts)


def chat(client: OpenAI, retriever: FAQRetriever, model: str, history: list[dict], user_message: str) -> str:
    # Retrieve relevant FAQ entries
    results = retriever.retrieve(user_message, top_k=3)
    context = build_context(results)

    # Build messages: system (with injected context) + conversation history + new user turn
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *history,
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="FAQ RAG Chatbot")
    parser.add_argument("--faq", default="faq.md", help="Path to the FAQ markdown file")
    parser.add_argument("--model", default=CHAT_MODEL, help="OpenAI chat model to use")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    print("Initializing FAQ RAG Chatbot...")
    retriever = FAQRetriever(args.faq, client)
    print(f"Ready! Using chat model: {args.model}")
    print("Type 'quit' or 'exit' to stop.\n")
    print("=" * 60)

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        answer = chat(client, retriever, args.model, history, user_input)
        print(f"\nBot: {answer}\n")

        # Keep last 6 turns (3 user + 3 assistant) to avoid token bloat
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        history = history[-6:]


if __name__ == "__main__":
    main()
