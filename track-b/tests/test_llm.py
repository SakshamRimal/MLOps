import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm_client import llm_client
from app.schemas import AssistantAnswer


async def main():
    try:
        result = await llm_client.chat_structured(
            "What is the capital of France?"
        )
        print("Raw dict:", result)
        validated = AssistantAnswer(**result)
        print("Validated object:", validated)
    except Exception as e:
        print(f"LLM call raised (expected if offline): {e}")


if __name__ == "__main__":
    result = llm_client.chat_structured(
        "What is the capital of France?"
    )
    print("Raw dict:", result)

    validated = AssistantAnswer(**result)
    print("Validated object:", validated)
    asyncio.run(main())