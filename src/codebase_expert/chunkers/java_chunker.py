# chunkers/java_chunker.py

from tree_sitter import Parser
from tree_sitter_languages import get_language
from .base import BaseChunker, get_node_text, ChunkingError
from .fallback_chunker import FallbackChunker
from typing import List

class JavaChunker(BaseChunker):
    """AST-based chunker for Java."""
    def __init__(self, max_chunk_size: int):
        super().__init__(max_chunk_size)
        self.language = get_language('java')
        self.parser = Parser()
        self.parser.set_language(self.language)
        # This query captures entire classes, interfaces, and enums as single chunks.
        self.query = self.language.query("""
        [
          (class_declaration) @chunk
          (interface_declaration) @chunk
          (enum_declaration) @chunk
        ]
        """)
        self.fallback_chunker = FallbackChunker(max_chunk_size)

    def chunk(self, content: str) -> List[str]:
        # The logic is identical to PythonChunker, can be refactored into a common base class
        # if we want to DRY it up later. For now, explicit is fine.
        try:
            content_bytes = content.encode('utf-8')
            tree = self.parser.parse(content_bytes)
            
            chunks = []
            last_end = 0
            
            captures = self.query.captures(tree.root_node)
            if not captures:
                return self.fallback_chunker.chunk(content)

            for node, _ in captures:
                if node.start_byte > last_end:
                    prologue = content_bytes[last_end:node.start_byte].decode('utf-8').strip()
                    if prologue:
                        chunks.extend(self.fallback_chunker.chunk(prologue))
                
                chunk_text = get_node_text(node, content_bytes)
                if len(chunk_text) > self.max_chunk_size:
                    chunks.extend(self.fallback_chunker.chunk(chunk_text))
                else:
                    chunks.append(chunk_text)
                
                last_end = node.end_byte

            if last_end < len(content_bytes):
                epilogue = content_bytes[last_end:].decode('utf-8').strip()
                if epilogue:
                    chunks.extend(self.fallback_chunker.chunk(epilogue))

            return chunks if chunks else self.fallback_chunker.chunk(content)
        except Exception as e:
            raise ChunkingError(f"Failed to parse Java code: {e}")