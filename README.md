# Codebase Expert - AI-Powered Code Understanding Tool

Transform any codebase into a searchable video memory that enables AI-powered code understanding, search, and Q&A. Works with any programming language or framework.

## 🚀 Features

- **Universal Compatibility**: Works with any codebase (Python, JavaScript, TypeScript, Go, Rust, Java, etc.)
- **Smart Detection**: Auto-detects project type and structure
- **Video Memory**: Converts code into efficient video format using QR codes
- **AI-Powered Chat**: Natural language Q&A about your codebase
- **Semantic Search**: Find code by concept, not just text matching
- **MCP Integration**: Use as a tool with Claude AI
- **Self-Contained**: Single file, handles dependencies automatically

## 📦 Installation

### Quick Start
```bash
# Download the script
curl -O https://raw.githubusercontent.com/yourusername/codebase-expert/main/codebase_expert.py
chmod +x codebase_expert.py

# Run it (installs dependencies automatically)
python codebase_expert.py --help
```

### Using pipx
```bash
pipx install codebase-expert
```

### Using uvx (for MCP)
```bash
uvx codebase_expert.py serve
```

## 🎯 Usage

### 1. Generate Knowledge Base
First, create the video memory of your codebase:
```bash
cd /path/to/your/project
python codebase_expert.py generate
```

This creates:
- `codebase_memory.mp4` - Video containing all your code
- `codebase_index.json` - Searchable index

Generation time depends on codebase size (usually 1-5 minutes).

### 2. Interactive Chat
Ask questions about your codebase:
```bash
python codebase_expert.py chat
```

Example questions:
- "How does the authentication system work?"
- "Where is the database connection handled?"
- "Explain the API routing structure"
- "What design patterns are used in this project?"

### 3. Quick Questions
One-off questions without entering chat mode:
```bash
python codebase_expert.py ask "What database ORM is used?"
python codebase_expert.py ask "How are environment variables loaded?"
```

### 4. Code Search
Search for specific patterns or concepts:
```bash
python codebase_expert.py search "error handling"
python codebase_expert.py search "websocket implementation" --top-k 10
```

### 5. MCP Server Mode
Integrate with Claude Desktop:

Add to Claude's config file:
```json
{
  "mcpServers": {
    "my-project-expert": {
      "command": "python",
      "args": ["/path/to/codebase_expert.py", "serve"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

Or using uvx:
```json
{
  "mcpServers": {
    "my-project-expert": {
      "command": "uvx",
      "args": ["codebase_expert.py", "serve"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Base Path
```bash
python codebase_expert.py generate --base-path /specific/project/path
python codebase_expert.py chat --base-path /specific/project/path
```

### Search Options
```bash
# Get more results
python codebase_expert.py search "async functions" --top-k 20

# Search and ask in one command
python codebase_expert.py ask "$(python codebase_expert.py search 'database models')"
```

## 📋 Supported File Types

The tool automatically processes:
- **Languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, C#, Ruby, PHP
- **Web**: HTML, CSS, SCSS, Vue, React, Svelte
- **Config**: JSON, YAML, TOML, INI, .env
- **Docs**: Markdown, reStructuredText, plain text
- **Build**: Dockerfile, Makefile, CMake, Gradle
- **Database**: SQL, Prisma schemas

## 🧠 How It Works

1. **Scanning**: Recursively scans your project, respecting .gitignore
2. **Chunking**: Splits large files into digestible chunks
3. **Encoding**: Converts code chunks into QR codes in video frames
4. **Indexing**: Creates semantic embeddings for intelligent search
5. **Querying**: Uses AI to understand and answer questions

## 🛡️ Privacy & Security

- **100% Local**: All processing happens on your machine
- **No Data Transmission**: Code never leaves your computer
- **Gitignore Respect**: Automatically excludes sensitive files
- **Configurable**: Add custom ignore patterns as needed

## ⚙️ Configuration

### Custom Ignore Patterns
Create a `.codebaseignore` file in your project root:
```
# Additional patterns to ignore
*.secret
private/
temp/
```

### Environment Variables
```bash
# Set OpenAI API key for enhanced chat (optional)
export OPENAI_API_KEY="your-key"

# Use different AI provider
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-key"
```

## 🔍 Examples by Project Type

### Node.js/React Project
```bash
cd my-react-app
python codebase_expert.py generate
python codebase_expert.py ask "How is Redux configured?"
python codebase_expert.py search "useState hooks"
```

### Python/Django Project
```bash
cd my-django-site
python codebase_expert.py generate
python codebase_expert.py ask "What are all the API endpoints?"
python codebase_expert.py search "model definitions"
```

### Go Microservice
```bash
cd my-go-service
python codebase_expert.py generate
python codebase_expert.py ask "How is dependency injection handled?"
python codebase_expert.py search "grpc handlers"
```

## 📊 Performance

- **Small Projects** (<1k files): ~30 seconds
- **Medium Projects** (1k-5k files): 1-3 minutes
- **Large Projects** (5k+ files): 3-10 minutes

Memory usage scales with codebase size. Approximately:
- 100MB codebase → 10MB video
- 1GB codebase → 100MB video

## 🐛 Troubleshooting

### "No code files found"
- Check you're in the right directory
- Verify file extensions are recognized
- Check if files are gitignored

### "Import error"
- Run the script again, it auto-installs dependencies
- Or manually: `pip install memvid mcp`

### "Memory error"
- For very large codebases, increase Python memory limit
- Consider excluding large generated folders

### MCP Connection Issues
- Ensure Claude Desktop is running
- Check the path in MCP config is absolute
- Verify Python is in PATH

## 🤝 Contributing

Contributions welcome! The tool is designed to be:
- Language agnostic
- Framework neutral
- Easy to extend

## 📄 License

MIT License - Use freely in personal and commercial projects.

## 🙏 Acknowledgments

Built with:
- [memvid](https://github.com/memvid/memvid) - Video memory encoding
- [MCP](https://modelcontextprotocol.org) - AI tool protocol
- [Sentence Transformers](https://sbert.net) - Semantic search

---

**Pro tip**: Add `alias code-expert='python /path/to/codebase_expert.py'` to your shell config for quick access!
