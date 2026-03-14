from llm.ollama import OllamaLLM
from agent.core import Agent
from agent.tools import registry
import logging
import os
from dotenv import load_dotenv

load_dotenv()

langfuse_api_key = os.getenv("LANGFUSE_PUBLIC_KEY")

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = f"""You are a helpful assistant with access to tools.
Available tools: {registry.descriptions()}
To use a tool, write: TOOL: tool_name(arg='value')
Otherwise, respond directly."""

def main():

    llm = OllamaLLM()
    agent = Agent(llm=llm, system=SYSTEM_PROMPT)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        response = agent(user_input)
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()
