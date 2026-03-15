from duckduckgo_search import DDGS
from utils.logger import Logger

logger = Logger(__name__)

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo. No API key required.

    Args:
        query: The search query string.
        max_results: Number of results to return (default 5).

    Returns:
        Formatted string of search results.
    """
    logger.info(f"[web_search] query='{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   URL: {r['href']}\n"
                f"   {r['body']}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        logger.error(f"[web_search] failed: {e}")
        return f"Search failed: {e}"