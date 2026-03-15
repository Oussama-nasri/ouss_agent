import subprocess
import sys
import tempfile
import os
from utils.logger import Logger

logger = Logger(__name__)

# Dangerous patterns to block before execution
BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess",
    "shutil", "__import__", "eval(", "exec(",
    "open(", "rmdir", "remove", "unlink",
]


def run_python(code: str, timeout: int = 10) -> str:
    """
    Execute Python code safely in a subprocess sandbox.

    Args:
        code:    Python code string to execute.
        timeout: Max seconds to allow (default 10).

    Returns:
        stdout output, or error message.

    Safety:
        - Runs in isolated subprocess (not current process)
        - Hard timeout kills runaway code
        - Blocks dangerous imports/builtins before running
    """
    logger.info(f"[run_python] executing {len(code)} chars of code")

    # Static safety check — block dangerous patterns
    lower_code = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lower_code:
            return f"Execution blocked: '{pattern}' is not allowed for safety reasons."

    # Write to temp file and run in subprocess
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            logger.info(f"[run_python] success, output: {output[:100]}")
            return output or "(no output)"
        else:
            error = result.stderr.strip()
            logger.warning(f"[run_python] error: {error[:200]}")
            return f"Error:\n{error}"

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout}s."
    except Exception as e:
        return f"Execution failed: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass