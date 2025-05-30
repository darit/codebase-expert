# Enhanced Codebase Expert - AI-Powered Code Understanding Tool (v1.1.0)

A powerful wrapper around [memvid](https://github.com/AwePhD/memvid) that transforms any codebase into a searchable video memory with enhanced context, enabling AI-powered code understanding, semantic search, and natural language Q&A. Works with any programming language or framework.

## 🚀 Features

- **Universal Compatibility**: Works with any codebase (Python, JavaScript, TypeScript, Go, Rust, Java, etc.)
- **Smart Detection**: Auto-detects project type and structure
- **Video Memory**: Converts code into efficient video format using memvid's QR code encoding
- **Enhanced Context**: Includes folder structure, git history, dependencies, and README content
- **AI-Powered Chat**: Natural language Q&A with LM Studio integration
- **Semantic Search**: Find code by concept using vector embeddings
- **Organized Output**: Files stored in dedicated folders with metadata
- **Shareable Packages**: Create zip files for easy distribution
- **MCP Integration**: Use as a tool with Claude Desktop
- **Self-Contained**: Single file, handles dependencies automatically

## 📦 Installation

### Quick Start
```bash
# Download the script
curl -O https://raw.githubusercontent.com/darit/codebase-expert/refs/heads/main/codebase_expert.py
chmod +x codebase_expert.py

# Run it (installs memvid and dependencies automatically)
python codebase_expert.py --help
```

### For MCP with Claude Desktop
```bash
# Add to Claude's config to use as an MCP tool
claude mcp add "Codebase Expert" /path/to/venv/bin/python /path/to/codebase_expert.py serve
```

## 🎯 Usage

### 1. Generate Knowledge Base
First, create the video memory of your codebase:
```bash
cd /path/to/your/project
python codebase_expert.py generate

# Or with custom output directory
python codebase_expert.py generate --output-dir ./my-codebase-memory

# Generate and create shareable zip
python codebase_expert.py generate --zip
```

This creates a folder `codebase-memory-{project_name}/` containing:
- `codebase_memory.mp4` - Video containing all your code with enhanced context
- `codebase_index.json` - Searchable vector index
- `codebase_index.faiss` - Vector embeddings for fast search
- `metadata.json` - Project statistics, git info, and folder structure

### 2. Interactive Chat with LM Studio
For the best experience, use with [LM Studio](https://lmstudio.ai/):
```bash
# Start LM Studio and load any chat model
# Then run:
python codebase_expert.py chat

# Custom LM Studio port
python codebase_expert.py chat --port 8080

# Without LM Studio (search-only mode)
python codebase_expert.py chat --no-lm
```

Chat commands:
- Ask questions naturally
- `/search <query>` - Direct codebase search
- `/context <topic>` - Get detailed context
- `/info` - Show project information and statistics
- `/clear` - Clear conversation history
- `/help` - Show commands

### 3. Quick Operations
```bash
# One-off question
python codebase_expert.py ask "What database ORM is used?"

# Direct search
python codebase_expert.py search "error handling" --top-k 10
```

### 4. MCP Server Mode
Add to Claude Desktop's config file:
```json
{
  "mcpServers": {
    "my-project-expert": {
      "command": "python",
      "args": ["/absolute/path/to/codebase_expert.py", "serve"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

## 🧠 How It Works

This tool leverages [memvid](https://github.com/AwePhD/memvid) to:

1. **Scan** your codebase respecting .gitignore patterns
2. **Extract Context** including folder structure, git history, dependencies, and README
3. **Chunk** files into manageable pieces with special context chunks
4. **Encode** chunks into QR codes within video frames using memvid
5. **Index** content with semantic embeddings for intelligent search
6. **Search** using vector similarity to find relevant code and context
7. **Chat** using LM Studio to provide natural language answers

### Enhanced Context (v1.1.0)
The tool now includes special context chunks at the beginning of the video:
- **Project Overview**: Name, type, file count, and general information
- **Folder Structure**: Visual tree representation of your codebase organization
- **Git History**: Recent commits, current branch, and top contributors
- **File Types Summary**: Distribution of programming languages and file types
- **README Content**: Automatically includes project documentation
- **Dependencies**: Extracts from package.json, requirements.txt, etc.

## 📋 Supported File Types

Automatically processes common code files:
- **Languages**: `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.cpp`, `.cs`, `.rb`, `.php`
- **Web**: `.html`, `.css`, `.scss`, `.vue`, `.jsx`, `.tsx`, `.svelte`
- **Config**: `.json`, `.yaml`, `.toml`, `.ini`, `.env.example`
- **Docs**: `.md`, `.rst`, `.txt`
- **Build**: `Dockerfile`, `Makefile`, `.gradle`

## ⚙️ Configuration

### Custom Ignore Patterns
The tool respects `.gitignore` and uses sensible defaults. Create `.codebaseignore` for additional patterns:
```
*.secret
private/
temp/
```

### LM Studio Integration
- Default port: 1234
- Supports any chat model
- Falls back to search-only mode if unavailable

## 📊 Performance

Thanks to memvid's efficient encoding:
- **Compression**: ~10:1 ratio (100MB code → 10MB video)
- **Speed**: Processes thousands of files in minutes
- **Search**: Instant vector-based retrieval
- **Enhanced Context**: Adds ~5-10% to video size but significantly improves answer quality

## 🔍 Example Workflow

```bash
# 1. Generate knowledge base with enhanced context
cd ~/projects/my-app
python codebase_expert.py generate --zip

# 2. Start interactive chat with LM Studio
python codebase_expert.py chat

# Example conversation:
You: How does the authentication system work?
Assistant: Based on the codebase, the authentication system uses JWT tokens...

You: What's the project structure?
Assistant: Based on the folder structure, this is a Node.js monorepo with...

You: /info
[Shows project statistics, git info, and metadata]

You: /search database migrations
[Shows relevant migration files]

You: /context API endpoints
[Provides detailed context about all API routes]
```

## 🐛 Troubleshooting

### "No code files found"
- Verify you're in the correct directory
- Check if files match supported extensions
- Review .gitignore patterns

### "LM Studio not available"
- Ensure LM Studio is running with a loaded model
- Check port configuration matches
- Use `--no-lm` for search-only mode

### "Import error"
- Script auto-installs memvid on first run
- Or manually: `pip install memvid requests`

## 🙏 Acknowledgments

This tool is a wrapper around:
- [memvid](https://github.com/AwePhD/memvid) - Core video memory encoding technology
- [sentence-transformers](https://sbert.net) - Semantic embeddings
- [MCP](https://modelcontextprotocol.io) - AI tool protocol

## 🆕 What's New in v1.1.0

### Enhanced Context for Better RAG Quality
- **Folder Structure**: Visual tree representation helps AI understand project organization
- **Git History**: Recent commits and contributors provide development context
- **Dependencies**: Automatic extraction from package.json, requirements.txt, etc.
- **README Integration**: Project documentation is automatically included
- **File Type Analysis**: Statistical overview of language distribution

### Improved Organization
- Generated files stored in `codebase-memory-{project_name}/` folders
- Clean separation between different projects
- Easy sharing with zip packages

### Better Chat Experience
- Suppressed progress bars and logging output
- New `/info` command for project statistics
- Cleaner, more readable interface

## 📄 License

MIT License - Use freely in personal and commercial projects.

---

**Pro tip**: Add `alias cx='python /path/to/codebase_expert.py'` to your shell config for quick access!
