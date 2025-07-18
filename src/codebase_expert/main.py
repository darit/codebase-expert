"""
Main CodebaseExpert class - extracted from the original codebase_expert.py

This module contains the core CodebaseExpert functionality with proper imports
for the new package structure.
"""

# For now, let's import the original class and create a wrapper
# This allows us to test the CLI structure without a full refactor
import sys
from pathlib import Path

# Add the project root to the path to import the original module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import the original CodebaseExpert class
    # Try local packaged version first, then fall back to root file
    try:
        # Use importlib to load from packaged version
        import importlib.util
        from pathlib import Path
        core_file = Path(__file__).parent / "codebase_expert_core.py"
        spec = importlib.util.spec_from_file_location("codebase_expert_core", core_file)
        codebase_expert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codebase_expert_module)
        OriginalCodebaseExpert = codebase_expert_module.CodebaseExpert
    except (FileNotFoundError, AttributeError):
        # Fall back to root file for local development
        spec = importlib.util.spec_from_file_location("codebase_expert_module", project_root / "codebase_expert.py")
        codebase_expert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codebase_expert_module)
        OriginalCodebaseExpert = codebase_expert_module.CodebaseExpert
    
    # Create a wrapper class that maintains the same interface
    class CodebaseExpert(OriginalCodebaseExpert):
        """
        Wrapper around the original CodebaseExpert class.
        
        This maintains backward compatibility while allowing us to use
        the new package structure.
        """
        pass

except ImportError as import_error:
    # Fallback for development
    _import_error_msg = str(import_error)
    
    class CodebaseExpert:
        def __init__(self, base_path=None, output_dir=None):
            self.base_path = base_path
            self.output_dir = output_dir
            print(f"Warning: Using fallback CodebaseExpert class. Import error: {_import_error_msg}")
        
        def generate_video(self, create_zip=False, video_only_zip=False):
            print(f"Generate video called with zip={create_zip}, video_only={video_only_zip}")
        
        def interactive_chat(self, use_lm_studio=True, lm_studio_url="http://localhost:1234"):
            print(f"Chat called with LM Studio: {use_lm_studio}, URL: {lm_studio_url}")
        
        def ask_question(self, question):
            return f"Mock answer for: {question}"
        
        def search_codebase(self, query, top_k=5):
            return f"Mock search results for: {query} (top {top_k})"
        
        async def run_mcp_server(self):
            print("Mock MCP server running...")
            import asyncio
            await asyncio.sleep(1)
            print("Mock MCP server stopped")

# Export the class
__all__ = ["CodebaseExpert"]