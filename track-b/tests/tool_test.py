import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm_client import llm_client
from app.tools import execute_tool, TOOL_DEFINITIONS


async def main():
    print(f"Registered tools ({len(TOOL_DEFINITIONS)}): {[t['function']['name'] for t in TOOL_DEFINITIONS]}")
    
    # Direct tool test
    calc_res = execute_tool("calculator", '{"expression": "0.15 * 340 + 7"}')
    print("Direct calculator result:", calc_res)
    
    time_res = execute_tool("get_current_time", "{}")
    print("Direct time result:", time_res)

    try:
        result = await llm_client.chat_with_tools("What is 15% of 340, and then add 7 to that?")
        print("Model answer:", result.get("answer"))
        print("Tools used:", result.get("tool_calls_made"))
    except Exception as e:
        print(f"LLM tool call raised (expected if offline): {e}")


if __name__ == "__main__":
    result = llm_client.chat_with_tools("What is 15% of 340, and then add 7 to that?")
    print("Answer:", result["answer"])
    print("Tools used:", result["tool_calls_made"])
    asyncio.run(main())