# 🤖 Ollama Agent

A production-grade AI agent running locally on your Ollama model,
with real tools: web search, file I/O, and Python execution.

## Architecture

```
my_agent/
├── agent/
│   ├── core.py        # ReAct loop — orchestration only
│   ├── memory.py      # Sliding window conversation history
│   └── prompts.py     # System prompt builder
├── llm/
│   ├── base.py        # Abstract interface (swap providers freely)
│   └── ollama.py      # Ollama implementation with retry
├── tools/
│   ├── registry.py    # Central tool registry + auto-docs
│   ├── web_search.py  # DuckDuckGo search (no API key)
│   ├── file_io.py     # Read/write/list in ./workspace/
│   └── code_exec.py   # Sandboxed Python execution
├── config/
│   └── settings.py    # All settings from .env
├── utils/
│   ├── logger.py      # Structured logging
│   └── retry.py       # Exponential backoff
├── tests/
│   └── test_agent.py  # Unit tests (LLM mocked)
└── main.py            # CLI entry point
```

## Quickstart

### 1. Prerequisites
Make sure Ollama is running:
```bash
ollama serve
ollama pull llama3.2   # or your preferred model
```

### 2. Install dependencies
```bash
cd my_agent
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env to set your model name if different
```

### 4. Run
```bash
python main.py
```

## Example Conversations

**Web search:**
```
You: What are the latest AI news today?
Agent: [searches web, summarizes results]
```

**File I/O:**
```
You: Write a poem about Python and save it to poem.txt
Agent: [writes poem, saves to workspace/poem.txt]
```

**Code execution:**
```
You: Calculate the first 10 Fibonacci numbers
Agent: [runs Python, returns results]
```

**Chained tools:**
```
You: Search for the top 3 Python frameworks, then save a summary to frameworks.txt
Agent: [searches → processes → writes file]
```

## CLI Commands

| Command     | Action                          |
|-------------|----------------------------------|
| `exit`      | Quit the agent                  |
| `clear`     | Reset conversation memory       |
| `history`   | Show current memory contents    |

## Running Tests

Tests run without Ollama (LLM is mocked):
```bash
pip install pytest
python -m pytest tests/ -v
```

## Adding New Tools

1. Create your function in `tools/your_tool.py` with a clear docstring
2. Register it in `tools/registry.py`:
```python
from tools.your_tool import your_function
registry.register("your_tool_name")(your_function)
```
That's it — the system prompt updates automatically.

## Swapping LLM Providers

Implement `BaseLLM` and swap in `main.py`:
```python
# From Ollama
from llm.ollama import OllamaLLM
llm = OllamaLLM()

# To OpenAI (just implement llm/openai.py)
from llm.openai import OpenAILLM
llm = OpenAILLM()
```