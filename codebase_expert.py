#!/usr/bin/env python3
"""
Codebase Expert - Universal tool for codebase knowledge
Works with any project to create searchable video memory

MCP Installation (for Claude Desktop):
    claude mcp add "Codebase Expert" python /path/to/codebase_expert.py serve

Standalone Usage:
    # Generate video memory
    python codebase_expert.py generate
    
    # Interactive chat
    python codebase_expert.py chat
    
    # Quick question
    python codebase_expert.py ask "How does authentication work?"
    
    # Search codebase
    python codebase_expert.py search "database implementation"
"""

import os
import sys
import json
import subprocess
import threading
import time
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import fnmatch

# Conditional imports
try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from memvid import MemvidEncoder, MemvidRetriever, MemvidChat
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Code generation constants
DEFAULT_IGNORE_PATTERNS = [
    '.git', '.gitignore', 'node_modules', '__pycache__', '*.pyc', '*.pyo', '*.pyd',
    '.DS_Store', '*.log', '*.out', 'coverage', 'dist', 'build', '.env', '.env.*',
    '*.swp', '*.swo', '*~', 'venv', 'myenv', '.vscode', '.idea', '*.mp4', '*.json',
    'codebase_memory.mp4', 'codebase_index.json', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', 'target', 'Cargo.lock', 'package-lock.json',
    'yarn.lock', 'pnpm-lock.yaml', '.next', '.nuxt', '.svelte-kit'
]

CODE_EXTENSIONS = [
    # Web
    '.ts', '.tsx', '.js', '.jsx', '.vue', '.svelte',
    # Backend
    '.py', '.java', '.go', '.rs', '.rb', '.php', '.cs', '.cpp', '.c', '.h',
    # Config
    '.yml', '.yaml', '.json', '.toml', '.ini', '.conf',
    # Web assets
    '.html', '.css', '.scss', '.sass', '.less',
    # Scripts
    '.sh', '.bash', '.zsh', '.ps1', '.bat',
    # Documentation
    '.md', '.rst', '.txt',
    # Database
    '.sql', '.prisma',
    # Docker
    '.dockerfile', 'Dockerfile', 'docker-compose.yml',
    # Other
    '.xml', '.gradle', '.cmake', 'Makefile', '.env.example'
]

class CodebaseExpert:
    """Main expert class that handles all functionality."""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.getcwd()
        self.project_name = self.detect_project_name()
        self.video_path = os.path.join(self.base_path, "codebase_memory.mp4")
        self.index_path = os.path.join(self.base_path, "codebase_index.json")
        self.retriever = None
        self.video_generation_in_progress = False
        self.video_generation_start_time = None
        
        # MCP server if available
        self.server = Server("Codebase Expert") if MCP_AVAILABLE else None
    
    def detect_project_name(self) -> str:
        """Detect project name from various sources."""
        # Try package.json
        package_json = os.path.join(self.base_path, "package.json")
        if os.path.exists(package_json):
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    return data.get('name', 'project')
            except:
                pass
        
        # Try setup.py or pyproject.toml
        for file in ['setup.py', 'pyproject.toml']:
            path = os.path.join(self.base_path, file)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        if 'name' in content:
                            # Simple extraction
                            import re
                            match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                            if match:
                                return match.group(1)
                except:
                    pass
        
        # Try Cargo.toml
        cargo_toml = os.path.join(self.base_path, "Cargo.toml")
        if os.path.exists(cargo_toml):
            try:
                with open(cargo_toml, 'r') as f:
                    content = f.read()
                    import re
                    match = re.search(r'name\s*=\s*"([^"]+)"', content)
                    if match:
                        return match.group(1)
            except:
                pass
        
        # Use directory name
        return os.path.basename(self.base_path)
    
    def ensure_dependencies(self):
        """Ensure required dependencies are installed."""
        if not MEMVID_AVAILABLE:
            print("Installing required dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "memvid"], check=True)
            if MCP_AVAILABLE is False:
                subprocess.run([sys.executable, "-m", "pip", "install", "mcp"], check=True)
            print("Dependencies installed. Please restart the script.")
            sys.exit(0)
    
    def video_exists(self) -> bool:
        """Check if video and index exist."""
        return os.path.exists(self.video_path) and os.path.exists(self.index_path)
    
    def initialize_memvid(self):
        """Initialize memvid components."""
        if not MEMVID_AVAILABLE:
            raise ImportError("Memvid not available")
        
        try:
            self.retriever = MemvidRetriever(self.video_path, self.index_path)
            logger.info("Memvid retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memvid: {e}")
            raise
    
    def detect_project_type(self) -> str:
        """Detect the type of project."""
        types = []
        
        # Check for various project files
        if os.path.exists(os.path.join(self.base_path, "package.json")):
            types.append("Node.js/JavaScript")
        if os.path.exists(os.path.join(self.base_path, "requirements.txt")) or \
           os.path.exists(os.path.join(self.base_path, "setup.py")) or \
           os.path.exists(os.path.join(self.base_path, "pyproject.toml")):
            types.append("Python")
        if os.path.exists(os.path.join(self.base_path, "Cargo.toml")):
            types.append("Rust")
        if os.path.exists(os.path.join(self.base_path, "go.mod")):
            types.append("Go")
        if os.path.exists(os.path.join(self.base_path, "pom.xml")) or \
           os.path.exists(os.path.join(self.base_path, "build.gradle")):
            types.append("Java")
        if os.path.exists(os.path.join(self.base_path, "Gemfile")):
            types.append("Ruby")
        if os.path.exists(os.path.join(self.base_path, ".csproj")) or \
           any(f.endswith('.sln') for f in os.listdir(self.base_path) if os.path.isfile(os.path.join(self.base_path, f))):
            types.append("C#/.NET")
        
        return " + ".join(types) if types else "general"
    
    # Video generation methods
    def read_gitignore(self, path: str) -> List[str]:
        """Read gitignore patterns."""
        gitignore_path = os.path.join(path, '.gitignore')
        patterns = []
        
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
        return patterns
    
    def should_ignore(self, file_path: str, ignore_patterns: List[str]) -> bool:
        """Check if file should be ignored."""
        file_name = os.path.basename(file_path)
        
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(file_name, pattern):
                return True
            
            if pattern.endswith('/'):
                if any(part == pattern[:-1] for part in file_path.split(os.sep)):
                    return True
            
            if fnmatch.fnmatch(file_path, pattern):
                return True
            
            parts = file_path.split(os.sep)
            for i in range(len(parts)):
                if fnmatch.fnmatch(parts[i], pattern):
                    return True
        
        return False
    
    def has_code_extension(self, file_path: str) -> bool:
        """Check if file has code extension."""
        return any(file_path.lower().endswith(ext) for ext in CODE_EXTENSIONS)
    
    def split_large_content(self, content: str, max_size: int = 2000) -> List[str]:
        """Split large content into chunks."""
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def extract_code_chunks(self, scan_path: str, relative_to: str) -> List[str]:
        """Extract code chunks from directory."""
        chunks = []
        ignore_patterns = DEFAULT_IGNORE_PATTERNS + self.read_gitignore(scan_path)
        
        total_files = 0
        for root, dirs, files in os.walk(scan_path):
            dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d), ignore_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, relative_to)
                
                if self.should_ignore(relative_path, ignore_patterns):
                    continue
                
                if not self.has_code_extension(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        if not content.strip():
                            continue
                        
                        total_files += 1
                        content_chunks = self.split_large_content(content)
                        
                        for i, chunk_content in enumerate(content_chunks):
                            if len(content_chunks) > 1:
                                chunk = f"=== {relative_path} (part {i+1}/{len(content_chunks)}) ===\n\n{chunk_content}\n"
                            else:
                                chunk = f"=== {relative_path} ===\n\n{chunk_content}\n"
                            
                            chunks.append(chunk)
                
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        print(f"Processed {total_files} files into {len(chunks)} chunks")
        return chunks
    
    def generate_video(self):
        """Generate the video from codebase."""
        self.ensure_dependencies()
        
        print(f"Scanning {self.project_name} codebase at: {self.base_path}")
        print("This may take a few minutes for large codebases...")
        
        # Extract all code chunks
        all_chunks = self.extract_code_chunks(self.base_path, self.base_path)
        
        print(f"\nTotal chunks collected: {len(all_chunks)}")
        
        if not all_chunks:
            print("No code files found!")
            print("Make sure you're in the right directory and have code files.")
            return False
        
        print("\nBuilding video memory...")
        encoder = MemvidEncoder()
        encoder.add_chunks(all_chunks)
        
        encoder.build_video(
            self.video_path,
            self.index_path,
            show_progress=True
        )
        
        print(f"\nVideo generated successfully!")
        print(f"Video: {self.video_path}")
        print(f"Index: {self.index_path}")
        
        total_size = sum(len(chunk) for chunk in all_chunks)
        print(f"\nSummary:")
        print(f"- Total chunks: {len(all_chunks)}")
        print(f"- Total size: {total_size / 1024 / 1024:.2f} MB")
        print(f"- Video size: {os.path.getsize(self.video_path) / 1024 / 1024:.2f} MB")
        print(f"- Compression ratio: {os.path.getsize(self.video_path) / total_size * 100:.1f}%")
        
        return True
    
    # Query methods
    def search_codebase(self, query: str, top_k: int = 5) -> str:
        """Search the codebase."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()
        
        results = self.retriever.search_with_metadata(query, top_k=top_k)
        
        response = f"Found {len(results)} results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            # Extract text and score from metadata
            chunk = result.get('text', str(result))
            score = result.get('score', 0.0)
            
            response += f"**Result {i}** (relevance: {score:.3f}):\n"
            response += f"```\n{chunk[:500]}...\n```\n\n"
        
        return response
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the codebase using vector search."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()
        
        # Use vector search to find relevant content
        results = self.retriever.search_with_metadata(question, top_k=5)
        
        if not results:
            return "No relevant information found."
        
        response = f"Found relevant information for: '{question}'\n\n"
        for i, result in enumerate(results, 1):
            # Extract text and score from metadata
            chunk = result.get('text', str(result))
            score = result.get('score', 0.0)
                
            if score > 0.5:  # Only include highly relevant results
                response += f"**Source {i}** (relevance: {score:.3f}):\n"
                response += f"```\n{chunk}\n```\n\n"
        
        return response
    
    def get_context(self, topic: str, max_tokens: int = 2000) -> str:
        """Get context for a topic."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()
        
        # Use search to find relevant chunks and concatenate them
        results = self.retriever.search_with_metadata(topic, top_k=10)
        
        if not results:
            return f"No context found for '{topic}'"
        
        context_parts = []
        total_chars = 0
        
        for result in results:
            chunk = result.get('text', '')
            score = result.get('score', 0.0)
            
            # Only include relevant results
            if score > 0.3:
                # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
                chunk_tokens = len(chunk) // 4
                if total_chars + len(chunk) < max_tokens * 4:
                    context_parts.append(chunk)
                    total_chars += len(chunk)
                else:
                    # Add partial chunk to fit within token limit
                    remaining_chars = (max_tokens * 4) - total_chars
                    if remaining_chars > 100:
                        context_parts.append(chunk[:remaining_chars] + "...")
                    break
        
        context = "\n\n---\n\n".join(context_parts)
        return f"Context for '{topic}':\n\n{context}"
    
    # CLI methods
    def interactive_chat(self):
        """Run interactive search session."""
        print(f"Codebase Expert Search - {self.project_name}")
        print("Type 'exit' or 'quit' to end the session")
        print("Commands: /search <query>, /help")
        print("-" * 50)
        
        if not self.video_exists():
            print("Knowledge base not found. Generating...")
            self.generate_video()
        
        self.initialize_memvid()
        
        while True:
            try:
                question = input("\nQuery: ").strip()
                if question.lower() in ['exit', 'quit']:
                    break
                
                if question == '/help':
                    print("\nCommands:")
                    print("  /search <query> - Search for specific code")
                    print("  /help - Show this help")
                    print("  exit/quit - Exit")
                    print("\nJust type to search the codebase.")
                    continue
                
                if question.startswith('/search '):
                    query = question[8:]
                else:
                    query = question
                
                print("\nSearching...")
                response = self.search_codebase(query)
                print(response)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        print("\nGoodbye!")
    
    # MCP Server methods
    async def run_mcp_server(self):
        """Run as MCP server."""
        if not MCP_AVAILABLE:
            print("MCP not available. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "mcp"], check=True)
            print("Please restart to run as MCP server.")
            return
        
        # Set up handlers
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="search_codebase",
                    description=f"Search the {self.project_name} codebase",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="ask_expert",
                    description=f"Ask about the {self.project_name} codebase",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"}
                        },
                        "required": ["question"]
                    }
                ),
                types.Tool(
                    name="get_context",
                    description="Get context for a topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "max_tokens": {"type": "integer", "default": 2000}
                        },
                        "required": ["topic"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
            if not self.video_exists():
                return [types.TextContent(
                    type="text",
                    text="Knowledge base not found. Please generate it first."
                )]
            
            try:
                if name == "search_codebase":
                    result = self.search_codebase(
                        arguments.get("query", ""),
                        arguments.get("top_k", 5)
                    )
                elif name == "ask_expert":
                    result = self.ask_question(arguments.get("question", ""))
                elif name == "get_context":
                    result = self.get_context(
                        arguments.get("topic", ""),
                        arguments.get("max_tokens", 2000)
                    )
                else:
                    result = f"Unknown tool: {name}"
                
                return [types.TextContent(type="text", text=result)]
                
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        
        # Run server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="Codebase Expert",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Codebase Expert - Universal tool for codebase knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate              # Generate knowledge base
  %(prog)s chat                  # Interactive chat
  %(prog)s ask "How does X work?" # Quick question
  %(prog)s search "pattern"      # Search codebase
  %(prog)s serve                 # Run as MCP server
        """
    )
    
    parser.add_argument('command', nargs='?', default='chat',
                       choices=['generate', 'chat', 'ask', 'search', 'serve'],
                       help='Command to run')
    parser.add_argument('query', nargs='*', help='Query for ask/search commands')
    parser.add_argument('--base-path', help='Override base directory (default: current directory)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of search results')
    
    args = parser.parse_args()
    
    # Create expert
    expert = CodebaseExpert(args.base_path)
    
    # Execute command
    if args.command == 'generate':
        expert.generate_video()
    
    elif args.command == 'chat':
        expert.interactive_chat()
    
    elif args.command == 'ask':
        if not args.query:
            print("Please provide a question")
            sys.exit(1)
        question = ' '.join(args.query)
        print(expert.ask_question(question))
    
    elif args.command == 'search':
        if not args.query:
            print("Please provide a search query")
            sys.exit(1)
        query = ' '.join(args.query)
        print(expert.search_codebase(query, args.top_k))
    
    elif args.command == 'serve':
        asyncio.run(expert.run_mcp_server())

if __name__ == "__main__":
    main()
