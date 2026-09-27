"""Run with: python -m fomo_brain"""

import os

from . import FomoBrain, ModelUnavailable, OllamaProvider


def main() -> None:
    brain = FomoBrain(OllamaProvider(
        model=os.getenv("FOMO_MODEL", "qwen3:0.6b"),
        base_url=os.getenv("FOMO_OLLAMA_URL", "http://127.0.0.1:11434"),
    ))
    print("FOMO AI local brain. Type /reset to clear history or /quit to exit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if text == "/reset":
            brain.reset()
            print("FOMO AI: Conversation reset.")
            continue
        if not text:
            continue
        try:
            print(f"FOMO AI: {brain.ask(text)}")
        except (ModelUnavailable, ValueError) as exc:
            print(f"FOMO AI error: {exc}")


if __name__ == "__main__":
    main()