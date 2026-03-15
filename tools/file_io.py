import os
from pathlib import Path
from utils.logger import Logger

logger = Logger(__name__)

# Safety: restrict all file ops to this working directory
WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    """Prevent path traversal attacks — always inside workspace/."""
    path = (WORKSPACE / filename).resolve()
    if not str(path).startswith(str(WORKSPACE)):
        raise ValueError(f"Access denied: '{filename}' is outside the workspace.")
    return path


def read_file(filename: str) -> str:
    """
    Read a file from the workspace directory.

    Args:
        filename: Name of the file to read (e.g. 'notes.txt').

    Returns:
        File contents as a string.
    """
    logger.info(f"[read_file] filename='{filename}'")
    try:
        path = _safe_path(filename)
        if not path.exists():
            return f"Error: '{filename}' does not exist in workspace."
        content = path.read_text(encoding="utf-8")
        logger.info(f"[read_file] read {len(content)} chars")
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the workspace directory.

    Args:
        filename: Name of the file to write (e.g. 'output.txt').
        content:  The text content to write.

    Returns:
        Confirmation message.
    """
    logger.info(f"[write_file] filename='{filename}', {len(content)} chars")
    try:
        path = _safe_path(filename)
        path.write_text(content, encoding="utf-8")
        return f"File '{filename}' written successfully ({len(content)} chars)."
    except Exception as e:
        return f"Error writing file: {e}"


def list_files() -> str:
    """
    List all files in the workspace directory.

    Returns:
        Newline-separated list of filenames.
    """
    logger.info("[list_files]")
    try:
        files = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
        if not files:
            return "Workspace is empty."
        return "\n".join(sorted(files))
    except Exception as e:
        return f"Error listing files: {e}"