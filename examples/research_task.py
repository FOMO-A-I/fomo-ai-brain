"""Run source-grounded research with a configured local checkpoint."""

from fomo.api.chat import ChatService
from fomo.brain.model import LocalTransformersBackend


if __name__ == "__main__":
    source = input("Paste a short source text (up to 4,000 characters): ").strip()
    result = ChatService(LocalTransformersBackend()).task({
        "prompt": "Summarize the supplied source and clearly distinguish facts from uncertainty.",
        "sources": [{"title": "Provided text", "content": source}],
    })
    print(result["answer"])