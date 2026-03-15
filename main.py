import sys
from llm.ollama      import OllamaLLM
from agent.core      import Agent
from agent.prompts   import build_system_prompt
from utils.logger    import Logger

logger = Logger(__name__)

BANNER = """
╔══════════════════════════════════════════╗
║          🤖  Ollama Agent CLI            ║
║  Tools: web_search · file_io · python   ║
║  Type 'exit' to quit · 'clear' to reset ║
╚══════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    try:
        llm   = OllamaLLM()
        agent = Agent(llm=llm, system=build_system_prompt())
    except Exception as e:
        print(f"\n❌ Failed to start: {e}")
        print("Make sure Ollama is running: `ollama serve`")
        sys.exit(1)

    print("✅ Agent ready! Ask me anything.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye! 👋")
            break

        if user_input.lower() == "clear":
            agent.memory.clear()
            print("🧹 Memory cleared.\n")
            continue

        if user_input.lower() == "history":
            messages = agent.memory.get_all()
            non_system = [m for m in messages if m["role"] != "system"]
            print(f"\n📜 {len(non_system)} messages in memory:\n")
            for m in non_system:
                role = "You" if m["role"] == "user" else "Agent"
                print(f"  [{role}]: {m['content'][:80]}...")
            print()
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            response = agent(user_input)
            print(response)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            print(f"❌ Error: {e}")
        print()


if __name__ == "__main__":
    main()