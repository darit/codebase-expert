# chunkers/javascript_chunker.py

from tree_sitter import Parser
from tree_sitter_languages import get_language
from .base import BaseChunker, get_node_text, ChunkingError
from .fallback_chunker import FallbackChunker
from typing import List

class JavaScriptChunker(BaseChunker):
    """AST-based chunker for JavaScript/TypeScript."""
    def __init__(self, max_chunk_size: int, language_name: str = 'javascript'):
        super().__init__(max_chunk_size)
        self.language = get_language(language_name)
        self.parser = Parser()
        self.parser.set_language(self.language)
        self.query = self.language.query("""
        [
          (function_declaration) @chunk
          (class_declaration) @chunk
          (lexical_declaration (arrow_function)) @chunk
          (export_statement) @chunk
        ]
        """)
        self.fallback_chunker = FallbackChunker(max_chunk_size)

    def chunk(self, content: str) -> List[str]:
        try:
            content_bytes = content.encode('utf-8')
            tree = self.parser.parse(content_bytes)
            
            chunks = []
            last_end = 0
            
            captures = self.query.captures(tree.root_node)
            if not captures:
                return self.fallback_chunker.chunk(content)

            # A simple way to handle nested captures is to use a set of seen nodes
            processed_node_ids = set()
            
            sorted_captures = sorted(captures, key=lambda c: c[0].start_byte)

            for node, _ in sorted_captures:
                if node.id in processed_node_ids:
                    continue
                
                # Mark this node and its children as processed
                q = [node]
                while q:
                    curr = q.pop(0)
                    processed_node_ids.add(curr.id)
                    q.extend(curr.children)

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
            raise ChunkingError(f"Failed to parse JavaScript/TypeScript code: {e}")