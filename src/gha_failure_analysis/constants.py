"""Common constants used across modules."""

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple character-based heuristic.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN
