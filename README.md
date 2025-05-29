# Codebase Expert - AI-Powered Code Understanding Tool

A powerful wrapper around [memvid](https://github.com/AwePhD/memvid) that transforms any codebase into a searchable video memory, enabling AI-powered code understanding, semantic search, and natural language Q&A. Works with any programming language or framework.

## 🚀 Features

- **Universal Compatibility**: Works with any codebase (Python, JavaScript, TypeScript, Go, Rust, Java, etc.)
- **Smart Detection**: Auto-detects project type and structure
- **Video Memory**: Converts code into efficient video format using memvid's QR code encoding
- **AI-Powered Chat**: Natural language Q&A with LM Studio integration
- **Semantic Search**: Find code by concept using vector embeddings
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
# See MCP Server Mode section below
```

## 🎯 Usage

### 1. Generate Knowledge Base
First, create the video memory of your codebase:
```bash
cd /path/to/your/project
python codebase_expert.py generate
```

This creates:
- `codebase_memory.mp4` - Video containing all your code (using memvid encoding)
- `codebase_index.json` - Searchable vector index

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
2. **Chunk** files into manageable pieces
3. **Encode** chunks into QR codes within video frames using memvid
4. **Index** content with semantic embeddings for intelligent search
5. **Search** using vector similarity to find relevant code
6. **Chat** using LM Studio to provide natural language answers

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

## 🔍 Example Workflow

```bash
# 1. Generate knowledge base
cd ~/projects/my-app
python codebase_expert.py generate

# 2. Start interactive chat with LM Studio
python codebase_expert.py chat

# Example conversation:
You: How does the authentication system work?
Assistant: Based on the codebase, the authentication system uses JWT tokens...

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

## 📄 License

MIT License - Use freely in personal and commercial projects.

---

**Pro tip**: Add `alias cx='python /path/to/codebase_expert.py'` to your shell config for quick access!
