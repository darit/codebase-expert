# chunkers/csharp_chunker.py

from tree_sitter import Parser
from tree_sitter_languages import get_language
from .base import BaseChunker, get_node_text, ChunkingError
from .fallback_chunker import FallbackChunker
from .java_chunker import JavaChunker
from typing import List

class CSharpChunker(JavaChunker):  # Inherits chunking logic from JavaChunker
    """AST-based chunker for C#."""
    def __init__(self, max_chunk_size: int):
        BaseChunker.__init__(self, max_chunk_size)  # Call grandparent init
        self.language = get_language('c_sharp')
        self.parser = Parser()
        self.parser.set_language(self.language)
        # Capturing namespaces can be tricky as they might span the whole file.
        # Focusing on type definitions is often more effective for chunking.
        self.query = self.language.query("""
        [
          (class_declaration) @chunk
          (interface_declaration) @chunk
          (struct_declaration) @chunk
          (enum_declaration) @chunk
          (delegate_declaration) @chunk
          (record_declaration) @chunk
        ]
        """)
        self.fallback_chunker = FallbackChunker(max_chunk_size)