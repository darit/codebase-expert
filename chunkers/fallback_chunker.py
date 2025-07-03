# chunkers/fallback_chunker.py

from typing import List
from .base import BaseChunker

class FallbackChunker(BaseChunker):
    """A chunker that falls back to line-based splitting with overlap."""

    def chunk(self, content: str) -> List[str]:
        return self._split_with_overlap(content.split('\n'), self.max_chunk_size)

    def _split_with_overlap(self, lines: List[str], max_size: int) -> List[str]:
        """Splits content with overlap for better context preservation."""
        chunks = []
        overlap_lines = 5
        current_chunk = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1
            if current_size + line_size > max_size and len(current_chunk) == 0:
                # Single line exceeds max_size, split it character-wise
                for i in range(0, len(line), max_size):
                    chunks.append(line[i:i + max_size])
                continue

            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                if len(current_chunk) > overlap_lines:
                    current_chunk = current_chunk[-overlap_lines:]
                    current_size = sum(len(l) + 1 for l in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks