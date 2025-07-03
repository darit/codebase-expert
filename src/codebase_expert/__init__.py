"""
Codebase Expert - Enhanced tool for codebase knowledge with AST chunking and RAG.

This package provides comprehensive codebase analysis capabilities including:
- AST-based code chunking for better understanding
- Enhanced RAG (Retrieval Augmented Generation) system
- MCP (Model Context Protocol) server for Claude Desktop integration
- Multiple interaction modes: generate, chat, ask, search, serve
"""

__version__ = "1.1.0"
__author__ = "Codebase Expert Contributors"

# Import main classes for easy access
from .main import CodebaseExpert

__all__ = ["CodebaseExpert", "__version__"]