from typing import Generator

CHAR_PER_TOKEN = 4

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> Generator[str, None, None]:
    """
    Split text into chunks of approximately `chunk_size` tokens (words/characters approximation)
    with `overlap`. Yields chunks one by one to save memory.
    """
    
    chunk_char_size = chunk_size * CHAR_PER_TOKEN
    overlap_char_size = overlap * CHAR_PER_TOKEN
    
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_char_size, text_len)
        
        # Try to find a sentence boundary or space to break at
        if end < text_len:
            # Look for the last space within the last 100 chars of the chunk to avoid splitting words
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (chunk_char_size * 0.8):
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        
        # Exit if we reached the end
        if end == text_len:
            break

        start = end - overlap_char_size
        if start < 0: # Should not happen but safety
            start = 0
        
        # Prevent infinite loop if overlap is too big or chunk is too small (not moving forward)
        if start >= end:
            start = end
