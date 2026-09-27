"""Ask an explicitly configured local checkpoint to draft website components."""

from fomo.api.chat import ChatService
from fomo.brain.model import LocalTransformersBackend


if __name__ == "__main__":
    service = ChatService(LocalTransformersBackend())
    result = service.chat([
        {"role": "user", "content": (
            "Draft a small accessible HTML/CSS landing page for FOMO AI. "
            "Explain which parts would still need human review."
        )}
    ])
    print(result["answer"])