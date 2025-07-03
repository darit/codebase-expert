"""
Modern CLI interface for Codebase Expert using Typer.

This module provides a clean, typed CLI interface that replaces the legacy argparse implementation.
Each command is implemented as a separate function with proper type hints and documentation.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

# Import the main CodebaseExpert class (will need to move this to main.py)
from .main import CodebaseExpert

# Create the main Typer application
app = typer.Typer(
    name="codebase-expert",
    help="Enhanced Codebase Expert - Universal tool for codebase knowledge with AST chunking and RAG",
    epilog="For more information, visit: https://github.com/darit/codebase-expert",
    no_args_is_help=True,
)


@app.command()
def generate(
    base_path: Annotated[Optional[str], typer.Option("--base-path", "-p", help="Override base directory (default: current directory)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", "-o", help="Output directory for generated files")] = None,
    zip_output: Annotated[bool, typer.Option("--zip", help="Create a zip package after generation")] = False,
    video_only_zip: Annotated[bool, typer.Option("--video-only-zip", help="Create zip with only the video file")] = False,
):
    """Generate the codebase knowledge base with video memory and search indices."""
    typer.echo("🚀 Generating codebase knowledge base...")
    
    expert = CodebaseExpert(base_path, output_dir)
    expert.generate_video(create_zip=zip_output, video_only_zip=video_only_zip)
    
    typer.echo("✅ Knowledge base generation completed!")


@app.command()
def chat(
    base_path: Annotated[Optional[str], typer.Option("--base-path", "-p", help="Override base directory (default: current directory)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", "-o", help="Output directory for generated files")] = None,
    port: Annotated[int, typer.Option("--port", help="LM Studio port")] = 1234,
    host: Annotated[str, typer.Option("--host", help="LM Studio host")] = "localhost",
    no_lm: Annotated[bool, typer.Option("--no-lm", help="Disable LM Studio integration")] = False,
):
    """Start an interactive chat session with the codebase expert."""
    typer.echo("💬 Starting interactive chat session...")
    
    expert = CodebaseExpert(base_path, output_dir)
    lm_studio_url = f"http://{host}:{port}"
    use_lm_studio = not no_lm
    
    expert.interactive_chat(use_lm_studio=use_lm_studio, lm_studio_url=lm_studio_url)


@app.command()
def ask(
    query: Annotated[List[str], typer.Argument(help="Question to ask about the codebase")],
    base_path: Annotated[Optional[str], typer.Option("--base-path", "-p", help="Override base directory (default: current directory)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", "-o", help="Output directory for generated files")] = None,
):
    """Ask a specific question about the codebase."""
    if not query:
        typer.echo("❌ Please provide a question", err=True)
        raise typer.Exit(1)
    
    question = ' '.join(query)
    typer.echo(f"🤔 Asking: {question}")
    
    expert = CodebaseExpert(base_path, output_dir)
    result = expert.ask_question(question)
    
    typer.echo("📝 Answer:")
    typer.echo(result)


@app.command()
def search(
    query: Annotated[List[str], typer.Argument(help="Search query for the codebase")],
    base_path: Annotated[Optional[str], typer.Option("--base-path", "-p", help="Override base directory (default: current directory)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", "-o", help="Output directory for generated files")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of search results")] = 5,
):
    """Search the codebase for specific patterns or content."""
    if not query:
        typer.echo("❌ Please provide a search query", err=True)
        raise typer.Exit(1)
    
    search_query = ' '.join(query)
    typer.echo(f"🔍 Searching for: {search_query}")
    
    expert = CodebaseExpert(base_path, output_dir)
    result = expert.search_codebase(search_query, top_k)
    
    typer.echo("📋 Search Results:")
    typer.echo(result)


@app.command()
def serve(
    base_path: Annotated[Optional[str], typer.Option("--base-path", "-p", help="Override base directory (default: current directory)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", "-o", help="Output directory for generated files")] = None,
):
    """Start the MCP (Model Context Protocol) server for Claude Desktop integration."""
    typer.echo("🌐 Starting MCP server for Claude Desktop...")
    typer.echo("📁 Working directory: " + (base_path or str(Path.cwd())))
    
    expert = CodebaseExpert(base_path, output_dir)
    
    try:
        asyncio.run(expert.run_mcp_server())
    except KeyboardInterrupt:
        typer.echo("\n👋 MCP server stopped")
    except Exception as e:
        typer.echo(f"❌ Error running MCP server: {e}", err=True)
        raise typer.Exit(1)


# Add version command
@app.command()
def version():
    """Show the version of codebase-expert."""
    from . import __version__
    typer.echo(f"codebase-expert version {__version__}")


if __name__ == "__main__":
    app()