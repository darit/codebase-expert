#!/usr/bin/env python3
"""
Enhanced Codebase Expert - Universal tool for codebase knowledge
Works with any project to create searchable video memory with better organization

MCP Installation (for Claude Desktop):
    The MCP server automatically works with the current directory (cwd) you specify:
    
    {
      "mcpServers": {
        "codebase-expert": {
          "command": "python",
          "args": ["/path/to/codebase_expert.py", "serve"],
          "cwd": "/path/to/any/project"  # MCP will analyze this directory
        }
      }
    }
    
    The MCP server will automatically generate video memory if it doesn't exist.

Standalone Usage:
    # Generate video memory with organized output
    python codebase_expert.py generate --output-dir ./codebase-memory
    
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
import shutil
import zipfile
import io
from contextlib import redirect_stdout, redirect_stderr

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
    '.git', '.gitignore', 'node_modules', 'vendor', '__pycache__', '*.pyc', '*.pyo', '*.pyd',
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
    """Enhanced expert class with better organization and features."""

    def __init__(self, base_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.base_path = base_path or os.getcwd()
        self.project_name = self.detect_project_name()

        # Output directory management
        if output_dir:
            # Use provided output directory
            self.output_dir = output_dir
        else:
            # Default to organized folder name
            self.output_dir = os.path.join(self.base_path, f"codebase-memory-{self.project_name}")

        self.video_path = os.path.join(self.output_dir, "codebase_memory.mp4")
        self.index_path = os.path.join(self.output_dir, "codebase_index.json")
        self.faiss_path = os.path.join(self.output_dir, "codebase_index.faiss")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")

        # Keep track of whether we're using the old location
        self._using_old_location = False

        # Fallback to old location if it exists (only when not explicitly setting output_dir)
        # and only for reading existing files, not for generation
        if not output_dir:
            old_video = os.path.join(self.base_path, "codebase_memory.mp4")
            old_index = os.path.join(self.base_path, "codebase_index.json")
            if os.path.exists(old_video) and os.path.exists(old_index):
                # Use old location for reading if new location doesn't have files
                if not os.path.exists(self.video_path):
                    self.video_path = old_video
                    self.index_path = old_index
                    self.output_dir = self.base_path
                    self._using_old_location = True

        self.retriever = None
        self.video_generation_in_progress = False
        self.video_generation_start_time = None

        # File watching for real-time updates
        self.file_watcher = None
        self.watch_enabled = False
        self.change_queue = []
        self.last_update_time = None

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

        # Suppress initialization messages
        import logging
        old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)

        try:
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                self.retriever = MemvidRetriever(self.video_path, self.index_path)
        except Exception as e:
            raise
        finally:
            logging.root.setLevel(old_level)

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

    def split_large_content(self, content: str, max_size: int = 600) -> List[str]:
        """Split large content into semantic chunks with improved handling."""
        if len(content) <= max_size:
            return [content]

        chunks = []
        lines = content.split('\n')

        # Detect file type for better chunking
        file_ext = self._detect_content_type(lines[:50])  # Check first 50 lines

        if file_ext in ['.py', '.java', '.go', '.rs', '.cs', '.cpp']:
            # Language-aware chunking for structured languages
            chunks = self._split_by_semantic_blocks(lines, max_size)
        else:
            # Fallback to line-based chunking with overlap
            chunks = self._split_with_overlap(lines, max_size)

        return chunks

    def _detect_content_type(self, lines: List[str]) -> str:
        """Detect content type from first few lines."""
        content_sample = '\n'.join(lines[:10])

        # Python
        if 'import ' in content_sample or 'from ' in content_sample or 'def ' in content_sample:
            return '.py'
        # JavaScript/TypeScript
        elif 'const ' in content_sample or 'function ' in content_sample or 'import {' in content_sample:
            return '.js'
        # Java
        elif 'package ' in content_sample or 'public class' in content_sample:
            return '.java'
        # Go
        elif 'package main' in content_sample or 'func ' in content_sample:
            return '.go'
        # Rust
        elif 'fn ' in content_sample or 'use ' in content_sample:
            return '.rs'

        return '.txt'

    def _split_by_semantic_blocks(self, lines: List[str], max_size: int) -> List[str]:
        """Split code by semantic blocks (functions, classes, etc.)."""
        chunks = []
        current_block = []
        current_size = 0
        indent_stack = [0]  # Track indentation levels

        for i, line in enumerate(lines):
            # Detect semantic boundaries
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Check if this is a new top-level block
            is_new_block = (
                    (stripped.startswith(('def ', 'class ', 'function ', 'func ', 'public ', 'private '))
                     and indent <= indent_stack[0]) or
                    (i > 0 and not stripped and current_size > max_size * 0.7)  # Empty line when chunk is getting large
            )

            if is_new_block and current_size > max_size * 0.5 and current_block:
                # Save current block
                chunks.append('\n'.join(current_block))
                current_block = []
                current_size = 0
                indent_stack = [indent]

            # Add line to current block
            current_block.append(line)
            current_size += len(line) + 1

            # Update indent tracking
            if stripped:
                if indent > indent_stack[-1]:
                    indent_stack.append(indent)
                elif indent < indent_stack[-1]:
                    while indent_stack and indent < indent_stack[-1]:
                        indent_stack.pop()

            # Force split if block is too large
            if current_size > max_size:
                chunks.append('\n'.join(current_block))
                current_block = []
                current_size = 0
                indent_stack = [0]

        if current_block:
            chunks.append('\n'.join(current_block))

        return chunks

    def _split_oversized_line(self, line: str, max_size: int) -> List[str]:
        """Split a single oversized line into character-based chunks"""
        chunks = []
        for i in range(0, len(line), max_size):
            chunks.append(line[i:i + max_size])
        return chunks

    def _split_with_overlap(self, lines: List[str], max_size: int) -> List[str]:
        """Split content with overlap for better context preservation."""
        chunks = []
        overlap_lines = 5  # Number of lines to overlap
        current_chunk = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1

            # Handle oversized single lines
            if current_size + line_size > max_size and len(current_chunk) == 0:
                # Single line exceeds max_size, split it character-wise  
                line_chunks = self._split_oversized_line(line, max_size - 50)  # Leave room for JSON overhead
                chunks.extend(line_chunks)
                continue

            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep last few lines for context
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

    def extract_code_chunks(self, scan_path: str, relative_to: str, max_chunk_size: int = 600) -> Tuple[List[str], List[str]]:
        """Extract code chunks from directory."""
        chunks = []
        ignore_patterns = DEFAULT_IGNORE_PATTERNS + self.read_gitignore(scan_path)

        total_files = 0
        file_list = []

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
                        file_list.append(relative_path)

                        # Check file size and handle accordingly
                        file_size = len(content)
                        if file_size > 50000:  # Large file (>50KB)
                            logger.info(f"Processing large file: {relative_path} ({file_size/1024:.1f}KB)")

                        content_chunks = self.split_large_content(content, max_size=max_chunk_size)

                        for i, chunk_content in enumerate(content_chunks):
                            if len(content_chunks) > 1:
                                # Add context about chunk position
                                chunk = f"=== {relative_path} (part {i+1}/{len(content_chunks)}) ===\n\n{chunk_content}\n"
                            else:
                                chunk = f"=== {relative_path} ===\n\n{chunk_content}\n"

                            chunks.append(chunk)

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        print(f"Processed {total_files} files into {len(chunks)} chunks")
        return chunks, file_list

    def generate_context_chunks(self, file_list: List[str]) -> List[str]:
        """Generate special context chunks for better RAG quality."""
        context_chunks = []

        # 1. Project Overview Chunk
        overview = f"""=== PROJECT OVERVIEW ===

Project Name: {self.project_name}
Project Type: {self.detect_project_type()}
Base Path: {self.base_path}
Total Files: {len(file_list)}

This is a comprehensive codebase index containing all source code, configuration files, and documentation.
"""
        context_chunks.append(overview)

        # 2. Folder Structure Chunk
        structure = f"""=== FOLDER STRUCTURE ===

{self.get_folder_structure()}

This folder structure represents the organization of the codebase.
"""
        context_chunks.append(structure)

        # 3. Git History Chunk
        git_info = self.get_git_info()
        if 'error' not in git_info:
            git_chunk = f"""=== GIT HISTORY ===

Current Branch: {git_info.get('current_branch', 'unknown')}
Remote Origin: {git_info.get('remote_origin', 'unknown')}

Recent Commits:
{chr(10).join(git_info.get('recent_commits', [])[:10])}

Top Contributors:
{chr(10).join(git_info.get('top_contributors', [])[:5])}
"""
            context_chunks.append(git_chunk)

        # 4. File Extensions Summary
        extensions = {}
        for file in file_list:
            ext = os.path.splitext(file)[1]
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1

        ext_summary = f"""=== FILE TYPES SUMMARY ===

File extensions in this codebase:
"""
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            ext_summary += f"\n{ext}: {count} files"
        context_chunks.append(ext_summary)

        # 5. README content if exists
        for readme_name in ['README.md', 'readme.md', 'README.txt', 'README']:
            readme_path = os.path.join(self.base_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_content = f.read()
                        if readme_content:
                            readme_chunk = f"""=== {readme_name} ===

{readme_content}
"""
                            context_chunks.append(readme_chunk)
                            break
                except:
                    pass

        # 6. Package/Dependency Information
        dep_info = self.get_dependency_info()
        if dep_info:
            dep_chunk = f"""=== DEPENDENCIES ===

{dep_info}
"""
            context_chunks.append(dep_chunk)

        return context_chunks

    def get_dependency_info(self) -> str:
        """Extract dependency information from various package files."""
        dep_info = []

        # package.json for Node.js
        package_json = os.path.join(self.base_path, 'package.json')
        if os.path.exists(package_json):
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    deps = data.get('dependencies', {})
                    dev_deps = data.get('devDependencies', {})

                    dep_info.append("Node.js Dependencies:")
                    for dep, version in list(deps.items())[:20]:
                        dep_info.append(f"  {dep}: {version}")
                    if len(deps) > 20:
                        dep_info.append(f"  ... and {len(deps) - 20} more")

                    if dev_deps:
                        dep_info.append("\nDev Dependencies:")
                        for dep, version in list(dev_deps.items())[:10]:
                            dep_info.append(f"  {dep}: {version}")
                        if len(dev_deps) > 10:
                            dep_info.append(f"  ... and {len(dev_deps) - 10} more")
            except:
                pass

        # requirements.txt for Python
        requirements = os.path.join(self.base_path, 'requirements.txt')
        if os.path.exists(requirements):
            try:
                with open(requirements, 'r') as f:
                    lines = f.readlines()
                    dep_info.append("\nPython Requirements:")
                    for line in lines[:20]:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dep_info.append(f"  {line}")
                    if len(lines) > 20:
                        dep_info.append(f"  ... and more")
            except:
                pass

        return "\n".join(dep_info) if dep_info else ""

    def get_folder_structure(self, max_depth: int = 3) -> str:
        """Generate a tree-like folder structure."""
        structure_lines = []

        def add_tree(path: str, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return

            try:
                items = sorted(os.listdir(path))
                # Filter out common ignore patterns
                items = [item for item in items if not any(
                    fnmatch.fnmatch(item, pattern) for pattern in ['.git', 'node_modules', '__pycache__', '*.pyc', 'venv', 'myenv']
                )]

                for i, item in enumerate(items[:20]):  # Limit items per folder
                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1

                    if os.path.isdir(item_path):
                        structure_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                        extension = "    " if is_last else "│   "
                        add_tree(item_path, prefix + extension, depth + 1)
                    else:
                        structure_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item}")

                if len(items) > 20:
                    structure_lines.append(f"{prefix}└── ... ({len(items) - 20} more items)")
            except PermissionError:
                pass

        structure_lines.append(f"{self.project_name}/")
        add_tree(self.base_path, "")
        return "\n".join(structure_lines)

    def _analyze_and_enhance_query(self, question: str) -> List[str]:
        """Analyze query type and generate enhanced search queries."""
        queries = [question]  # Always include original

        # Detect architectural patterns
        architectural_keywords = [
            'architecture', 'design', 'structure', 'pattern', 'how does',
            'how do', 'implementation', 'workflow', 'system', 'component',
            'integration', 'relationship', 'interaction'
        ]

        question_lower = question.lower()
        is_architectural = any(keyword in question_lower for keyword in architectural_keywords)

        if is_architectural:
            # Extract key concepts from the question
            concepts = self._extract_concepts(question)

            # Generate variations
            for concept in concepts:
                # Add implementation-focused query
                queries.append(f"{concept} implementation")
                queries.append(f"{concept} class interface")
                queries.append(f"{concept} module structure")

            # Add overview queries
            queries.append("PROJECT OVERVIEW architecture")
            queries.append("FOLDER STRUCTURE organization")

        # Handle generic searches
        if len(question.split()) <= 3:  # Short, potentially generic query
            # Expand with common patterns
            base_terms = question.split()
            for term in base_terms:
                queries.append(f"class {term}")
                queries.append(f"function {term}")
                queries.append(f"module {term}")
                queries.append(f"{term} configuration")

        return queries[:10]  # Limit to prevent over-searching

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for query enhancement."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during',
            'how', 'does', 'do', 'what', 'where', 'when', 'why', 'is', 'are',
            'was', 'were', 'been', 'being', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that'
        }

        words = text.lower().split()
        concepts = []

        # Extract meaningful words
        for word in words:
            cleaned = word.strip('.,?!;:\'"')
            if cleaned and len(cleaned) > 2 and cleaned not in stop_words:
                concepts.append(cleaned)

        # Also extract bigrams for compound concepts
        for i in range(len(words) - 1):
            w1, w2 = words[i].strip('.,?!;:\'"'), words[i+1].strip('.,?!;:\'"')
            if w1 not in stop_words and w2 not in stop_words:
                concepts.append(f"{w1} {w2}")

        return concepts

    def _enhance_search_query(self, query: str) -> List[str]:
        """Enhance search query with contextual variations."""
        queries = [query]  # Original query

        # Handle camelCase and snake_case
        # Convert camelCase to space-separated
        import re
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
        if camel_split != query:
            queries.append(camel_split.lower())

        # Convert snake_case to space-separated
        snake_split = query.replace('_', ' ')
        if snake_split != query:
            queries.append(snake_split)

        # Add common programming patterns
        if len(query.split()) == 1:  # Single word query
            word = query.lower()
            # Common prefixes/suffixes
            queries.extend([
                f"get{query}", f"set{query}", f"{query}Handler",
                f"{query}Manager", f"{query}Service", f"{query}Controller",
                f"handle{query}", f"process{query}", f"create{query}"
            ])

            # File patterns
            queries.extend([
                f"{word}.py", f"{word}.js", f"{word}.ts",
                f"{word}.jsx", f"{word}.tsx", f"{word}.go"
            ])

        # Limit queries to prevent over-searching
        return list(dict.fromkeys(queries))[:8]

    def _deduplicate_and_rank_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate and rank search results by relevance."""
        seen_content = {}
        ranked_results = []

        for result in results:
            chunk = result.get('text', '')
            # Create content fingerprint
            content_key = chunk[:300].strip()  # First 300 chars

            if content_key in seen_content:
                # Update score if this is a better match
                existing_idx = seen_content[content_key]
                if result.get('score', 0) > ranked_results[existing_idx].get('score', 0):
                    ranked_results[existing_idx] = result
            else:
                seen_content[content_key] = len(ranked_results)
                ranked_results.append(result)

        # Sort by score
        return sorted(ranked_results, key=lambda x: x.get('score', 0), reverse=True)

    def _get_search_suggestions(self, query: str) -> str:
        """Provide helpful suggestions for failed searches."""
        suggestions = ["💡 Search suggestions:"]

        # Check if it's a very generic term
        generic_terms = ['config', 'handler', 'manager', 'service', 'util', 'helper', 'controller']
        if query.lower() in generic_terms:
            suggestions.append(f"- Try being more specific: '{query} authentication' or '{query} database'")
            suggestions.append(f"- Search for actual implementations: 'class {query}' or 'function {query}'")

        # Suggest file search
        suggestions.append(f"- Search for files: '{query}.py' or '{query}.js'")

        # Suggest function/class search
        suggestions.append(f"- Search for definitions: 'def {query}' or 'class {query}'")

        # Suggest exploring project structure
        suggestions.append("- Use '/info' to see project structure")
        suggestions.append("- Try '/context overview' for project overview")

        return "\n".join(suggestions)

    def get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        git_info = {}

        try:
            # Check if it's a git repo
            subprocess.run(['git', 'rev-parse', '--git-dir'],
                           cwd=self.base_path, capture_output=True, check=True)

            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'],
                                    cwd=self.base_path, capture_output=True, text=True)
            git_info['current_branch'] = result.stdout.strip()

            # Get recent commits
            result = subprocess.run(['git', 'log', '--oneline', '-10'],
                                    cwd=self.base_path, capture_output=True, text=True)
            git_info['recent_commits'] = result.stdout.strip().split('\n')

            # Get remote origin
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                    cwd=self.base_path, capture_output=True, text=True)
            git_info['remote_origin'] = result.stdout.strip()

            # Get contributors
            result = subprocess.run(['git', 'shortlog', '-sn', '--all'],
                                    cwd=self.base_path, capture_output=True, text=True)
            git_info['top_contributors'] = result.stdout.strip().split('\n')[:5]

        except (subprocess.CalledProcessError, FileNotFoundError):
            git_info['error'] = 'Not a git repository'

        return git_info

    def generate_metadata(self, file_list: List[str], chunks: List[str]) -> Dict[str, Any]:
        """Generate enhanced metadata for the codebase."""
        total_size = sum(len(chunk) for chunk in chunks)

        metadata = {
            "project_name": self.project_name,
            "project_type": self.detect_project_type(),
            "base_path": self.base_path,
            "generation_date": datetime.now().isoformat(),
            "statistics": {
                "total_files": len(file_list),
                "total_chunks": len(chunks),
                "total_size_mb": total_size / 1024 / 1024,
                "unique_extensions": list(set(os.path.splitext(f)[1] for f in file_list if os.path.splitext(f)[1]))
            },
            "file_list": file_list[:100],  # First 100 files
            "folder_structure": self.get_folder_structure(),
            "git_info": self.get_git_info(),
            "memvid_version": "latest",
            "expert_version": "1.1.0"
        }

        return metadata

    def create_package(self, include_video_only: bool = False) -> str:
        """Create a zip package with all necessary files."""
        package_name = f"{self.project_name}_codebase_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = os.path.join(self.base_path, package_name)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Always include the video
            if os.path.exists(self.video_path):
                zipf.write(self.video_path, os.path.basename(self.video_path))

            if not include_video_only:
                # Include index files
                if os.path.exists(self.index_path):
                    zipf.write(self.index_path, os.path.basename(self.index_path))
                if os.path.exists(self.faiss_path):
                    zipf.write(self.faiss_path, os.path.basename(self.faiss_path))
                if os.path.exists(self.metadata_path):
                    zipf.write(self.metadata_path, os.path.basename(self.metadata_path))

                # Include a README
                readme_content = f"""# {self.project_name} Codebase Memory

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project Type: {self.detect_project_type()}

## Contents
- codebase_memory.mp4: Video containing encoded codebase
- codebase_index.json: Search index for memvid
- codebase_index.faiss: Vector embeddings (if available)
- metadata.json: Project metadata and statistics

## Usage
Place all files in the same directory and use with codebase_expert.py:

```bash
python codebase_expert.py chat --base-path /path/to/extracted/files
```

Or use directly with memvid:
```python
from memvid import MemvidRetriever
retriever = MemvidRetriever("codebase_memory.mp4", "codebase_index.json")
results = retriever.search_with_metadata("your query", top_k=5)
```
"""
                zipf.writestr("README.txt", readme_content)

        return package_path

    def generate_video(self, create_zip: bool = False, video_only_zip: bool = False):
        """Generate the video from codebase with enhanced organization."""
        self.ensure_dependencies()
        
        # Define QR generation parameters for clarity and easy tuning.
        # A QR code (v40, 'L' correction) holds ~2953 bytes. Based on verification,
        # JSON overhead is significant (~14KB for 1.8KB content). We need much smaller chunks.
        MAX_CHUNK_CONTENT_SIZE = 200  # Reduced from 300 for JSON overhead safety

        # If we're using old location for reading, reset paths for generation
        if self._using_old_location:
            self.output_dir = os.path.join(self.base_path, f"codebase-memory-{self.project_name}")
            self.video_path = os.path.join(self.output_dir, "codebase_memory.mp4")
            self.index_path = os.path.join(self.output_dir, "codebase_index.json")
            self.faiss_path = os.path.join(self.output_dir, "codebase_index.faiss")
            self.metadata_path = os.path.join(self.output_dir, "metadata.json")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"🚀 Scanning {self.project_name} codebase at: {self.base_path}")
        print(f"📁 Output directory: {self.output_dir}")
        print("⏳ This may take a few minutes for large codebases...")

        # Extract all code chunks
        all_chunks, file_list = self.extract_code_chunks(self.base_path, self.base_path, MAX_CHUNK_CONTENT_SIZE)

        print(f"\n📊 Total chunks collected: {len(all_chunks)}")

        # Add special context chunks
        print("🔍 Adding enhanced context...")
        context_chunks = self.generate_context_chunks(file_list)
        all_chunks = context_chunks + all_chunks
        print(f"📊 Total chunks with context: {len(all_chunks)}")

        if not all_chunks:
            print("❌ No code files found!")
            print("Make sure you're in the right directory and have code files.")
            return False

        # Generate metadata
        metadata = self.generate_metadata(file_list, all_chunks)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n🎬 Building video memory...")
        
        # --- Verification Step ---
        # Log the largest chunk's JSON size to confirm it's within safe limits.
        # The memvid library adds its own overhead ('chunk_id', etc.), but the content
        # is the primary driver of size.
        if all_chunks:
            max_json_size = 0
            for chunk in all_chunks:
                # Get size of our chunk when serialized to JSON in bytes
                json_size = len(json.dumps(chunk).encode('utf-8'))
                if json_size > max_json_size:
                    max_json_size = json_size
            
            # QR v40 'L' capacity is 2953 bytes.
            print(f"📏 Max chunk JSON size: {max_json_size} bytes (QR limit: ~2950 bytes)")
        # --- End Verification Step ---
        
        # Get default configuration and only override QR error correction for better capacity
        from memvid.config import get_default_config
        config = get_default_config()
        config['QR_ERROR_CORRECTION'] = 'L'  # Low error correction for maximum data capacity
        
        encoder = MemvidEncoder(config=config)
        encoder.add_chunks(all_chunks)

        encoder.build_video(
            self.video_path,
            self.index_path,
            show_progress=True
        )

        # Check if FAISS index was created
        if not os.path.exists(self.faiss_path):
            # memvid might save it with a different name, try to find it
            possible_faiss = os.path.join(self.output_dir, "codebase_index.faiss")
            if not os.path.exists(possible_faiss):
                possible_faiss = os.path.join(os.path.dirname(self.index_path),
                                              os.path.splitext(os.path.basename(self.index_path))[0] + ".faiss")
            if os.path.exists(possible_faiss) and possible_faiss != self.faiss_path:
                shutil.copy2(possible_faiss, self.faiss_path)

        print(f"\n✅ Video generated successfully!")
        print(f"📹 Video: {self.video_path}")
        print(f"🔍 Index: {self.index_path}")
        print(f"📊 Metadata: {self.metadata_path}")

        total_size = sum(len(chunk) for chunk in all_chunks)
        print(f"\n📈 Summary:")
        print(f"- Total files: {len(file_list)}")
        print(f"- Total chunks: {len(all_chunks)}")
        print(f"- Total size: {total_size / 1024 / 1024:.2f} MB")
        print(f"- Video size: {os.path.getsize(self.video_path) / 1024 / 1024:.2f} MB")
        print(f"- Compression ratio: {os.path.getsize(self.video_path) / total_size * 100:.1f}%")

        # Create zip package if requested
        if create_zip:
            print(f"\n📦 Creating shareable package...")
            package_path = self.create_package(include_video_only=video_only_zip)
            print(f"✅ Package created: {package_path}")
            print(f"   Size: {os.path.getsize(package_path) / 1024 / 1024:.2f} MB")

        print(f"\n💡 Next steps:")
        print(f"1. Run interactive chat: python {os.path.basename(__file__)} chat")
        print(f"2. Share the folder: {self.output_dir}")
        if create_zip:
            print(f"3. Or share the zip: {os.path.basename(package_path)}")

        return True

    # Query methods
    def search_codebase(self, query: str, top_k: int = 5) -> str:
        """Search the codebase with enhanced contextual understanding."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()

        # Enhance query for better results
        enhanced_queries = self._enhance_search_query(query)

        # Suppress all output during search including logging
        import logging
        old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)

        all_results = []
        try:
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                # Search with enhanced queries
                for enhanced_query in enhanced_queries:
                    results = self.retriever.search_with_metadata(enhanced_query, top_k=max(3, top_k//2))
                    for result in results:
                        result['query_used'] = enhanced_query
                    all_results.extend(results)
        finally:
            logging.root.setLevel(old_level)

        # Deduplicate and rank results
        unique_results = self._deduplicate_and_rank_results(all_results)

        if not unique_results:
            # Provide helpful suggestions for generic queries
            suggestions = self._get_search_suggestions(query)
            return f"No results found for '{query}'.\n\n{suggestions}"

        response = f"Found {len(unique_results[:top_k])} results for '{query}':\n\n"
        for i, result in enumerate(unique_results[:top_k], 1):
            # Extract text and score from metadata
            chunk = result.get('text', str(result))
            score = result.get('score', 0.0)
            query_used = result.get('query_used', query)

            response += f"**Result {i}** (relevance: {score:.3f})"
            if query_used != query:
                response += f" [searched: '{query_used}']"
            response += f":\n```\n{chunk[:500]}...\n```\n\n"

        return response

    def ask_question(self, question: str) -> str:
        """Ask a question about the codebase using enhanced vector search with query analysis."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()

        # Analyze question type and enhance query
        enhanced_queries = self._analyze_and_enhance_query(question)

        # Use vector search to find relevant content, suppress all output
        import logging
        old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)

        all_results = []
        try:
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                # Search with multiple query variations
                for query in enhanced_queries:
                    results = self.retriever.search_with_metadata(query, top_k=3)
                    all_results.extend(results)
        finally:
            logging.root.setLevel(old_level)

        # Deduplicate and sort by relevance
        seen_chunks = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.get('score', 0), reverse=True):
            chunk = result.get('text', '')
            chunk_preview = chunk[:200]  # Use first 200 chars for deduplication
            if chunk_preview not in seen_chunks:
                seen_chunks.add(chunk_preview)
                unique_results.append(result)

        if not unique_results:
            return "No relevant information found."

        response = f"Found relevant information for: '{question}'\n\n"
        for i, result in enumerate(unique_results[:5], 1):
            # Extract text and score from metadata
            chunk = result.get('text', str(result))
            score = result.get('score', 0.0)

            if score > 0.3:  # Lower threshold for architectural questions
                response += f"**Source {i}** (relevance: {score:.3f}):\n"
                response += f"```\n{chunk}\n```\n\n"

        return response

    def get_context(self, topic: str, max_tokens: int = 2000) -> str:
        """Get context for a topic."""
        if not self.retriever:
            if not self.video_exists():
                return "Knowledge base not found. Run 'generate' first."
            self.initialize_memvid()

        # Use search to find relevant chunks and concatenate them, suppress all output
        import logging
        old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)

        try:
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                results = self.retriever.search_with_metadata(topic, top_k=10)
        finally:
            logging.root.setLevel(old_level)

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

    def get_project_info(self) -> str:
        """Get project information from metadata."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            info = f"📊 Project: {metadata['project_name']}\n"
            info += f"🔧 Type: {metadata['project_type']}\n"
            info += f"📅 Generated: {metadata['generation_date']}\n"
            info += f"📁 Files: {metadata['statistics']['total_files']}\n"
            info += f"📦 Chunks: {metadata['statistics']['total_chunks']}\n"
            info += f"💾 Size: {metadata['statistics']['total_size_mb']:.2f} MB\n"

            return info
        else:
            return "No metadata found. Generate the knowledge base first."

    # File watching methods
    def start_file_watcher(self):
        """Start watching for file changes in the codebase."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class CodebaseChangeHandler(FileSystemEventHandler):
                def __init__(self, expert):
                    self.expert = expert
                    self.ignore_patterns = DEFAULT_IGNORE_PATTERNS

                def on_modified(self, event):
                    if not event.is_directory and self._should_process(event.src_path):
                        self.expert._handle_file_change('modified', event.src_path)

                def on_created(self, event):
                    if not event.is_directory and self._should_process(event.src_path):
                        self.expert._handle_file_change('created', event.src_path)

                def on_deleted(self, event):
                    if not event.is_directory and self._should_process(event.src_path):
                        self.expert._handle_file_change('deleted', event.src_path)

                def _should_process(self, path):
                    # Check if file should be processed
                    relative_path = os.path.relpath(path, self.expert.base_path)
                    return (not self.expert.should_ignore(relative_path, self.ignore_patterns)
                            and self.expert.has_code_extension(path))

            self.file_watcher = Observer()
            event_handler = CodebaseChangeHandler(self)
            self.file_watcher.schedule(event_handler, self.base_path, recursive=True)
            self.file_watcher.start()
            self.watch_enabled = True
            logger.info(f"Started file watcher for {self.base_path}")

            # Start background thread to process changes
            import threading
            self._change_processor_thread = threading.Thread(target=self._process_change_queue)
            self._change_processor_thread.daemon = True
            self._change_processor_thread.start()

        except ImportError:
            logger.warning("watchdog not installed. Install with: pip install watchdog")
            return False
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            return False

        return True

    def stop_file_watcher(self):
        """Stop watching for file changes."""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
            self.watch_enabled = False
            logger.info("Stopped file watcher")

    def _handle_file_change(self, change_type: str, file_path: str):
        """Handle a file change event."""
        relative_path = os.path.relpath(file_path, self.base_path)
        change_info = {
            'type': change_type,
            'path': relative_path,
            'timestamp': datetime.now().isoformat()
        }

        self.change_queue.append(change_info)
        logger.debug(f"File {change_type}: {relative_path}")

    def _process_change_queue(self):
        """Process queued file changes in background."""
        import time

        while self.watch_enabled:
            if self.change_queue and len(self.change_queue) >= 10:
                # Process changes in batch
                changes = self.change_queue[:10]
                self.change_queue = self.change_queue[10:]

                # Update the index incrementally
                self._update_index_incrementally(changes)

            time.sleep(5)  # Check every 5 seconds

    def _update_index_incrementally(self, changes: List[Dict]):
        """Update the search index incrementally based on file changes."""
        if not self.retriever:
            return

        try:
            # This is a placeholder for incremental update logic
            # In a real implementation, you would:
            # 1. Extract chunks from changed files
            # 2. Update the vector embeddings
            # 3. Update the search index
            # 4. Save the updated index

            logger.info(f"Processing {len(changes)} file changes...")

            # For now, mark that an update is needed
            self.last_update_time = datetime.now()

            # Notify user in interactive mode
            if hasattr(self, '_interactive_mode'):
                print(f"\n📝 {len(changes)} files changed. Consider regenerating the knowledge base.")

        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def get_recent_changes(self, limit: int = 10) -> str:
        """Get recent file changes."""
        if not self.change_queue:
            return "No recent changes detected."

        changes = self.change_queue[-limit:]
        response = f"Recent file changes ({len(changes)} total):\n\n"

        for change in changes:
            response += f"- {change['type']}: {change['path']} at {change['timestamp']}\n"

        if self.last_update_time:
            response += f"\nLast index update: {self.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}"

        return response

    # CLI methods
    def interactive_chat(self, use_lm_studio=True, lm_studio_url="http://localhost:1234"):
        """Run interactive chat session with optional LM Studio integration."""
        # ANSI color codes
        class Colors:
            HEADER = '\033[95m'
            BLUE = '\033[94m'
            CYAN = '\033[96m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'

        # LM Studio integration
        lm_chat = None
        conversation_history = []

        if use_lm_studio:
            try:
                import requests
                # Test LM Studio connection
                try:
                    response = requests.get(f"{lm_studio_url}/v1/models", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('data', [])
                        if models:
                            lm_chat = {
                                'url': lm_studio_url,
                                'model': models[0]['id'],
                                'system_prompt': f"""You are a helpful AI assistant that specializes in analyzing the {self.project_name} codebase.
You have access to a searchable knowledge base of the current project's code.
When answering questions, you search the codebase and provide accurate, relevant information.
Be concise but thorough in your responses."""
                            }
                            print(f"{Colors.GREEN}✓ LM Studio connected - Model: {lm_chat['model']}{Colors.ENDC}")
                        else:
                            print(f"{Colors.YELLOW}⚠ LM Studio connected but no model loaded{Colors.ENDC}")
                except:
                    print(f"{Colors.YELLOW}⚠ LM Studio not available - using search-only mode{Colors.ENDC}")
            except ImportError:
                print(f"{Colors.YELLOW}⚠ requests library not available - using search-only mode{Colors.ENDC}")

        # Print header
        print(f"\n{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}║     Enhanced Codebase Expert - {self.project_name:<22} ║{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}╚══════════════════════════════════════════════════════╝{Colors.ENDC}")

        # Show project info if available
        if os.path.exists(self.metadata_path):
            print(f"\n{self.get_project_info()}")

        print(f"\n{Colors.CYAN}Commands:{Colors.ENDC}")
        print("  • Type your questions about the codebase")
        print("  • /search <query> - Direct codebase search")
        print("  • /context <topic> - Get detailed context")
        print("  • /info - Show project information")
        print("  • /watch - Start file change monitoring")
        print("  • /unwatch - Stop file change monitoring")
        print("  • /changes - Show recent file changes")
        print("  • /clear - Clear conversation history")
        print("  • /help - Show this help")
        print("  • exit or quit - Exit")
        print(f"\n{Colors.YELLOW}{'─' * 55}{Colors.ENDC}\n")

        if not self.video_exists():
            print("Knowledge base not found. Generating...")
            self.generate_video()

        self.initialize_memvid()

        while True:
            try:
                question = input(f"{Colors.GREEN}You: {Colors.ENDC}").strip()
                if question.lower() in ['exit', 'quit']:
                    break

                if question == '/help':
                    print(f"\n{Colors.CYAN}Commands:{Colors.ENDC}")
                    print("  • Type questions naturally")
                    print("  • /search <query> - Direct search")
                    print("  • /context <topic> - Get context")
                    print("  • /info - Show project info")
                    print("  • /watch - Start file monitoring")
                    print("  • /unwatch - Stop file monitoring")
                    print("  • /changes - Show recent changes")
                    print("  • /clear - Clear chat history")
                    print("  • /help - Show this help")
                    print("  • exit/quit - Exit")
                    continue

                if question == '/info':
                    print(f"\n{self.get_project_info()}")
                    continue

                if question == '/clear':
                    conversation_history = []
                    print(f"{Colors.CYAN}Conversation history cleared.{Colors.ENDC}")
                    continue

                if question == '/watch':
                    if self.start_file_watcher():
                        print(f"{Colors.GREEN}✓ File monitoring started{Colors.ENDC}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Could not start file monitoring (install watchdog: pip install watchdog){Colors.ENDC}")
                    continue

                if question == '/unwatch':
                    self.stop_file_watcher()
                    print(f"{Colors.CYAN}File monitoring stopped{Colors.ENDC}")
                    continue

                if question == '/changes':
                    print(f"\n{self.get_recent_changes()}")
                    continue

                if question.startswith('/search '):
                    query = question[8:]
                    print(f"\n{Colors.BOLD}Search Results:{Colors.ENDC}")
                    response = self.search_codebase(query, top_k=5)
                    print(response)
                    continue

                if question.startswith('/context '):
                    topic = question[9:]
                    print(f"\n{Colors.BOLD}Context:{Colors.ENDC}")
                    response = self.get_context(topic, max_tokens=3000)
                    print(response)
                    continue

                # Process with LM Studio if available
                print(f"\n{Colors.BLUE}Assistant: {Colors.ENDC}", end='', flush=True)

                if lm_chat and use_lm_studio:
                    try:
                        import requests
                        # Search codebase first (search_codebase already suppresses output)
                        search_results = self.search_codebase(question, top_k=3)

                        # Build prompt with context
                        prompt = f"""Based on the following search results from the codebase, please answer the user's question.

User Question: {question}

Codebase Search Results:
{search_results.replace('**', '').replace('```', '')}

Please provide a helpful and accurate answer based on the search results above."""

                        # Add to conversation
                        conversation_history.append({"role": "user", "content": prompt})

                        # Build messages
                        messages = [{"role": "system", "content": lm_chat['system_prompt']}] + conversation_history

                        # Call LM Studio
                        response = requests.post(
                            f"{lm_chat['url']}/v1/chat/completions",
                            json={
                                "messages": messages,
                                "temperature": 0.7,
                                "max_tokens": -1,
                                "stream": False
                            },
                            timeout=60
                        )

                        if response.status_code == 200:
                            data = response.json()
                            answer = data['choices'][0]['message']['content']
                            print(answer)

                            # Update history with original question
                            conversation_history[-1]["content"] = question
                            conversation_history.append({"role": "assistant", "content": answer})
                        else:
                            print(f"{Colors.RED}LM Studio error. Using search-only mode.{Colors.ENDC}")
                            print(self.ask_question(question))
                    except Exception as e:
                        print(f"{Colors.RED}Error with LM Studio: {e}{Colors.ENDC}")
                        print("\nFalling back to search-only mode:")
                        print(self.ask_question(question))
                else:
                    # Use simple search mode
                    response = self.ask_question(question)
                    print(response)

                print()  # Empty line for readability

            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'exit' to quit.{Colors.ENDC}")
                continue
            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")

        print(f"\n{Colors.CYAN}Thank you for using Codebase Expert!{Colors.ENDC}")

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
                ),
                types.Tool(
                    name="get_project_info",
                    description="Get project information and statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="generate_codebase_video",
                    description="Generate or regenerate the codebase video memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force": {"type": "boolean", "default": False, "description": "Force regeneration even if video exists"}
                        }
                    }
                ),
                types.Tool(
                    name="monitor_changes",
                    description="Start or stop monitoring file changes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"}
                        },
                        "required": ["action"]
                    }
                ),
                types.Tool(
                    name="get_recent_changes",
                    description="Get recent file changes detected by the monitor",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10, "description": "Number of recent changes to show"}
                        }
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
            # Auto-generate video if it doesn't exist
            if not self.video_exists() and name != "get_project_info":
                try:
                    # Generate video in background
                    result_text = f"Generating knowledge base for {self.project_name}. This may take a few minutes...\n"
                    result_text += "The codebase video is being generated. Please try again in a moment."

                    # Start generation in a separate thread
                    import threading
                    def generate_async():
                        try:
                            self.generate_video()
                        except Exception as e:
                            logger.error(f"Failed to generate video: {e}")

                    if not self.video_generation_in_progress:
                        self.video_generation_in_progress = True
                        self.video_generation_start_time = time.time()
                        thread = threading.Thread(target=generate_async)
                        thread.daemon = True
                        thread.start()
                    else:
                        elapsed = time.time() - self.video_generation_start_time
                        result_text = f"Video generation in progress... ({int(elapsed)}s elapsed)"

                    return [types.TextContent(type="text", text=result_text)]
                except Exception as e:
                    return [types.TextContent(
                        type="text",
                        text=f"Failed to start video generation: {str(e)}"
                    )]

            try:
                if name == "search_codebase":
                    if not self.retriever:
                        self.initialize_memvid()
                    result = self.search_codebase(
                        arguments.get("query", ""),
                        arguments.get("top_k", 5)
                    )
                elif name == "ask_expert":
                    if not self.retriever:
                        self.initialize_memvid()
                    result = self.ask_question(arguments.get("question", ""))
                elif name == "get_context":
                    if not self.retriever:
                        self.initialize_memvid()
                    result = self.get_context(
                        arguments.get("topic", ""),
                        arguments.get("max_tokens", 2000)
                    )
                elif name == "get_project_info":
                    result = self.get_project_info()
                elif name == "generate_codebase_video":
                    force = arguments.get("force", False)
                    if force or not self.video_exists():
                        # Start generation
                        if not self.video_generation_in_progress:
                            self.video_generation_in_progress = True
                            self.video_generation_start_time = time.time()

                            # Generate synchronously for MCP
                            try:
                                self.generate_video()
                                result = f"Successfully generated video memory for {self.project_name}"
                                self.video_generation_in_progress = False
                            except Exception as e:
                                self.video_generation_in_progress = False
                                result = f"Failed to generate video: {str(e)}"
                        else:
                            elapsed = time.time() - self.video_generation_start_time
                            result = f"Video generation already in progress... ({int(elapsed)}s elapsed)"
                    else:
                        result = f"Video already exists for {self.project_name}. Use force=true to regenerate."
                elif name == "monitor_changes":
                    action = arguments.get("action", "status")
                    if action == "start":
                        if self.start_file_watcher():
                            result = f"Started monitoring file changes in {self.project_name}"
                        else:
                            result = "Failed to start file monitoring. Install watchdog: pip install watchdog"
                    elif action == "stop":
                        self.stop_file_watcher()
                        result = "Stopped monitoring file changes"
                    else:  # status
                        if self.watch_enabled:
                            result = f"File monitoring is active. {len(self.change_queue)} changes queued."
                        else:
                            result = "File monitoring is not active"
                elif name == "get_recent_changes":
                    limit = arguments.get("limit", 10)
                    result = self.get_recent_changes(limit)
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
        description="Enhanced Codebase Expert - Universal tool for codebase knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate                    # Generate knowledge base in organized folder
  %(prog)s generate --zip              # Generate and create shareable zip
  %(prog)s generate --video-only-zip   # Create zip with just the video
  %(prog)s chat                        # Interactive chat with LM Studio
  %(prog)s chat --no-lm                # Chat without LM Studio (search only)
  %(prog)s chat --port 8080            # Use custom LM Studio port
  %(prog)s ask "How does X work?"      # Quick question
  %(prog)s search "pattern"            # Search codebase
  %(prog)s serve                       # Run as MCP server
        """
    )

    parser.add_argument('command', nargs='?', default='chat',
                        choices=['generate', 'chat', 'ask', 'search', 'serve'],
                        help='Command to run')
    parser.add_argument('query', nargs='*', help='Query for ask/search commands')
    parser.add_argument('--base-path', help='Override base directory (default: current directory)')
    parser.add_argument('--output-dir', help='Output directory for generated files')
    parser.add_argument('--top-k', type=int, default=5, help='Number of search results')

    # Generation options
    parser.add_argument('--zip', action='store_true',
                        help='Create a zip package after generation')
    parser.add_argument('--video-only-zip', action='store_true',
                        help='Create zip with only the video file')

    # LM Studio options
    parser.add_argument('--port', type=int, default=1234,
                        help='LM Studio port (default: 1234)')
    parser.add_argument('--host', default='localhost',
                        help='LM Studio host (default: localhost)')
    parser.add_argument('--no-lm', action='store_true',
                        help='Disable LM Studio integration')

    args = parser.parse_args()

    # Create expert
    expert = CodebaseExpert(args.base_path, args.output_dir)

    # Execute command
    if args.command == 'generate':
        expert.generate_video(create_zip=args.zip, video_only_zip=args.video_only_zip)

    elif args.command == 'chat':
        # Build LM Studio URL
        lm_studio_url = f"http://{args.host}:{args.port}"
        use_lm_studio = not args.no_lm
        expert.interactive_chat(use_lm_studio=use_lm_studio, lm_studio_url=lm_studio_url)

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
