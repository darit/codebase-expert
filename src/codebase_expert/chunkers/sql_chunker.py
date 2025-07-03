# chunkers/sql_chunker.py

from tree_sitter import Parser
from tree_sitter_languages import get_language
from .base import BaseChunker, get_node_text, ChunkingError
from .fallback_chunker import FallbackChunker
from typing import List

class SqlChunker(BaseChunker):
    """AST-based chunker for SQL."""
    def __init__(self, max_chunk_size: int):
        super().__init__(max_chunk_size)
        self.language = get_language('sql')
        self.parser = Parser()
        self.parser.set_language(self.language)
        # This query aims to split on a statement-by-statement basis
        self.query = self.language.query("""
        (statement) @chunk
        """)
        self.fallback_chunker = FallbackChunker(max_chunk_size)

    def chunk(self, content: str) -> List[str]:
        # Same logic as Python/Java chunkers
        try:
            content_bytes = content.encode('utf-8')
            tree = self.parser.parse(content_bytes)
            
            chunks = []
            last_end = 0
            
            captures = self.query.captures(tree.root_node)
            if not captures:
                return self.fallback_chunker.chunk(content)

            for node, _ in captures:
                chunk_text = get_node_text(node, content_bytes).strip()
                if chunk_text:
                    chunks.append(chunk_text)

            return chunks if chunks else self.fallback_chunker.chunk(content)
        except Exception as e:
            raise ChunkingError(f"Failed to parse SQL code: {e}")