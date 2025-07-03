# chunkers/factory.py

from pathlib import Path
from .base import BaseChunker
from .fallback_chunker import FallbackChunker
from .python_chunker import PythonChunker
from .javascript_chunker import JavaScriptChunker
from .java_chunker import JavaChunker
from .csharp_chunker import CSharpChunker
from .sql_chunker import SqlChunker

def get_chunker(file_path: str, max_chunk_size: int) -> BaseChunker:
    """Factory function to get the appropriate chunker for a file."""
    extension = Path(file_path).suffix.lower()
    
    if extension == '.py':
        return PythonChunker(max_chunk_size)
    elif extension in ['.js', '.jsx', '.mjs']:
        return JavaScriptChunker(max_chunk_size, 'javascript')
    elif extension in ['.ts', '.tsx']:
        return JavaScriptChunker(max_chunk_size, 'typescript')
    elif extension == '.java':
        return JavaChunker(max_chunk_size)
    elif extension == '.cs':
        return CSharpChunker(max_chunk_size)
    elif extension == '.sql':
        return SqlChunker(max_chunk_size)
    else:
        # For any other file type, use the reliable line-based splitter.
        return FallbackChunker(max_chunk_size)