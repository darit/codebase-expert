# RAG System Enhancement Technical Specification
## Codebase Expert Tool - Advanced Implementation Guide

### Executive Summary

This document outlines a comprehensive enhancement strategy for the existing RAG (Retrieval-Augmented Generation) system in the codebase expert tool. The proposed improvements focus on advanced chunking strategies, embedding model optimization, hybrid retrieval systems, and intelligent context assembly to significantly improve code search and understanding capabilities.

### Table of Contents

1. [Current System Analysis](#current-system-analysis)
2. [Areas for Improvement](#areas-for-improvement)
3. [Implementation Architecture](#implementation-architecture)
4. [Technical Specifications](#technical-specifications)
5. [Implementation Phases](#implementation-phases)
6. [Performance and Scalability](#performance-and-scalability)
7. [Testing and Validation](#testing-and-validation)

---

## Current System Analysis

### Architecture Overview

The existing codebase expert tool implements a basic RAG system with the following components:

**Core Components:**
- **Document Ingestion**: File-based content extraction and preprocessing
- **Chunking Strategy**: Line-based chunking with configurable overlap
- **Embedding Generation**: Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings
- **Vector Storage**: In-memory or file-based vector index
- **Retrieval Engine**: Semantic similarity search using cosine distance
- **Video Memory**: Innovative QR code-based storage for compressed knowledge representation

**Current Strengths:**
- Functional semantic search capabilities
- Video memory innovation for knowledge compression
- Basic overlap handling for context preservation
- MCP (Model Context Protocol) integration

**Current Limitations:**
- Generic embedding model not optimized for code
- Line-based chunking ignores code structure
- Single retrieval strategy (semantic only)
- Limited context assembly intelligence
- No code-specific metadata utilization

### Current Chunking Strategy Analysis

```python
# Current approach (conceptual)
def line_based_chunking(content, chunk_size=512, overlap=50):
    lines = content.split('\n')
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

**Issues with Current Approach:**
- Splits functions/classes arbitrarily
- Loses semantic boundaries
- Equal treatment of code vs. comments vs. documentation
- No consideration for language-specific syntax

---

## Areas for Improvement

### 1. Advanced Chunking Strategy

#### AST-Based Semantic Chunking

Replace line-based chunking with Abstract Syntax Tree (AST) analysis for structure-aware segmentation.

**Technical Approach:**
```python
import ast
import tree_sitter
from typing import List, Dict, Any

class CodeChunker:
    def __init__(self, language: str):
        self.language = language
        self.parser = self._init_parser(language)
    
    def semantic_chunk(self, code: str) -> List[Dict[str, Any]]:
        """
        Create chunks based on semantic code boundaries
        """
        tree = self.parser.parse(bytes(code, 'utf8'))
        chunks = []
        
        # Extract functions, classes, and modules as primary chunks
        for node in self._walk_ast(tree.root_node):
            if node.type in ['function_definition', 'class_definition', 'module']:
                chunk = {
                    'content': self._extract_node_content(code, node),
                    'type': node.type,
                    'name': self._extract_name(node),
                    'start_line': node.start_point[0],
                    'end_line': node.end_point[0],
                    'metadata': self._extract_metadata(node)
                }
                chunks.append(chunk)
        
        return chunks
```

**Language-Specific Strategies:**

1. **JavaScript/TypeScript**
    - Function declarations and expressions
    - Class definitions and methods
    - Export statements and modules
    - React component boundaries

2. **Python**
    - Function and class definitions
    - Module-level code blocks
    - Decorator preservation

3. **Java/C#**
    - Class and interface definitions
    - Method boundaries
    - Package/namespace awareness

4. **SQL**
    - Statement boundaries (SELECT, INSERT, UPDATE, DELETE)
    - Stored procedure definitions
    - View and table definitions

#### Adaptive Chunk Sizing

```python
class AdaptiveChunker:
    def __init__(self):
        self.size_limits = {
            'function': (100, 800),      # min, max tokens
            'class': (200, 1500),
            'module': (500, 2000),
            'documentation': (50, 500)
        }
    
    def determine_chunk_size(self, content_type: str, complexity: int) -> int:
        """
        Dynamic sizing based on content type and complexity
        """
        base_min, base_max = self.size_limits[content_type]
        
        # Adjust based on complexity metrics
        complexity_factor = min(complexity / 10.0, 2.0)  # Cap at 2x
        
        optimal_size = int(base_min + (base_max - base_min) * complexity_factor)
        return min(optimal_size, base_max)
```

### 2. Embedding Model Enhancement

#### Code-Specific Embedding Models Evaluation

**Recommended Models for Implementation:**

1. **Microsoft CodeBERT** (`microsoft/codebert-base`)
    - Pre-trained on code-text pairs
    - Excellent for code-natural language understanding
    - 768-dimensional embeddings

2. **Salesforce CodeT5** (`Salesforce/codet5-base`)
    - Encoder-decoder architecture
    - Strong performance on code generation tasks
    - Multi-language support

3. **OpenAI Code Embeddings** (`text-embedding-ada-002`)
    - High-quality embeddings
    - API-based (cost consideration)
    - 1536-dimensional embeddings

4. **Code-specific Sentence Transformers**
    - Fine-tuned models for code similarity
    - Local deployment friendly

**Implementation Strategy:**
```python
class CodeEmbeddingManager:
    def __init__(self, model_type: str = 'codebert'):
        self.models = {
            'codebert': self._init_codebert(),
            'codet5': self._init_codet5(),
            'ada002': self._init_openai(),
            'multilingual': self._init_multilingual()
        }
        self.active_model = self.models[model_type]
    
    def embed_code_chunk(self, chunk: Dict[str, Any]) -> np.ndarray:
        """
        Generate embeddings with context awareness
        """
        content_type = chunk.get('type', 'unknown')
        
        if content_type in ['function', 'method']:
            return self._embed_function(chunk)
        elif content_type in ['class', 'interface']:
            return self._embed_class(chunk)
        else:
            return self._embed_generic(chunk)
    
    def _embed_function(self, chunk: Dict[str, Any]) -> np.ndarray:
        """
        Function-specific embedding with signature emphasis
        """
        content = chunk['content']
        signature = self._extract_signature(content)
        docstring = self._extract_docstring(content)
        
        # Weighted combination of signature, docstring, and body
        combined_text = f"SIGNATURE: {signature}\nDOCUMENTATION: {docstring}\nCODE: {content}"
        return self.active_model.encode(combined_text)
```

#### Multi-Modal Embedding Strategy

```python
class MultiModalEmbedding:
    def __init__(self):
        self.code_model = CodeBertModel()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.fusion_weights = {'code': 0.7, 'text': 0.3}
    
    def generate_embedding(self, chunk: Dict[str, Any]) -> np.ndarray:
        """
        Combine code and text embeddings with adaptive weighting
        """
        code_ratio = self._calculate_code_ratio(chunk['content'])
        
        # Adjust fusion weights based on content composition
        code_weight = min(0.9, 0.5 + code_ratio * 0.4)
        text_weight = 1.0 - code_weight
        
        code_emb = self.code_model.encode(chunk['content'])
        text_emb = self.text_model.encode(chunk['content'])
        
        # Weighted fusion
        combined = code_weight * code_emb + text_weight * text_emb
        return combined / np.linalg.norm(combined)
```

### 3. Hybrid Retrieval System

#### Semantic + Keyword Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np
from scipy.sparse import csr_matrix

class HybridRetriever:
    def __init__(self, semantic_weight: float = 0.7):
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        
        # Initialize components
        self.semantic_index = None  # Vector index
        self.keyword_index = None   # BM25 index
        self.chunks = []
        
    def build_indices(self, chunks: List[Dict[str, Any]]):
        """
        Build both semantic and keyword indices
        """
        # Semantic index (existing)
        embeddings = [self.embed_chunk(chunk) for chunk in chunks]
        self.semantic_index = self._build_vector_index(embeddings)
        
        # Keyword index (BM25)
        tokenized_chunks = [self._tokenize_chunk(chunk) for chunk in chunks]
        self.keyword_index = BM25Okapi(tokenized_chunks)
        
        self.chunks = chunks
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results
        """
        # Semantic search
        query_embedding = self.embed_query(query)
        semantic_scores = self.semantic_index.similarity_search(
            query_embedding, top_k=top_k*2
        )
        
        # Keyword search
        tokenized_query = self._tokenize_query(query)
        keyword_scores = self.keyword_index.get_scores(tokenized_query)
        
        # Combine scores using weighted fusion
        combined_scores = self._fuse_scores(
            semantic_scores, keyword_scores, 
            self.semantic_weight, self.keyword_weight
        )
        
        # Rank and return top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
```

#### Query Expansion for Code Queries

```python
class CodeQueryExpander:
    def __init__(self):
        self.synonyms = {
            'function': ['method', 'procedure', 'def', 'func'],
            'class': ['object', 'type', 'interface', 'struct'],
            'variable': ['var', 'let', 'const', 'field', 'property'],
            'loop': ['for', 'while', 'iterate', 'foreach'],
            'condition': ['if', 'when', 'check', 'validate']
        }
        
        self.code_patterns = {
            'error_handling': ['try', 'catch', 'except', 'finally', 'throw'],
            'async': ['async', 'await', 'promise', 'future', 'callback'],
            'database': ['query', 'select', 'insert', 'update', 'delete', 'sql']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate expanded query variations for better retrieval
        """
        expanded_queries = [query]
        
        # Add synonym variations
        for term, synonyms in self.synonyms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_queries.append(query.replace(term, synonym))
        
        # Add pattern-based expansions
        for pattern, terms in self.code_patterns.items():
            if any(term in query.lower() for term in terms):
                pattern_query = f"{query} {pattern}"
                expanded_queries.append(pattern_query)
        
        return expanded_queries
```

### 4. Context Window Optimization

#### Intelligent Context Assembly

```python
class ContextAssembler:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.relevance_threshold = 0.7
        
    def assemble_context(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Intelligently assemble context based on relevance and relationships
        """
        # Score and rank chunks
        scored_chunks = self._score_chunks(query, retrieved_chunks)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(scored_chunks)
        
        # Select optimal chunk combination
        selected_chunks = self._select_optimal_chunks(
            scored_chunks, dependency_graph, self.max_tokens
        )
        
        # Arrange chunks in logical order
        ordered_chunks = self._order_chunks_logically(selected_chunks)
        
        return self._format_context(ordered_chunks)
    
    def _score_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score chunks based on multiple factors
        """
        scored = []
        for chunk in chunks:
            score = {
                'semantic_similarity': self._semantic_score(query, chunk),
                'keyword_match': self._keyword_score(query, chunk),
                'code_complexity': self._complexity_score(chunk),
                'freshness': self._freshness_score(chunk),
                'completeness': self._completeness_score(chunk)
            }
            
            # Weighted combination
            final_score = (
                0.4 * score['semantic_similarity'] +
                0.3 * score['keyword_match'] +
                0.1 * score['code_complexity'] +
                0.1 * score['freshness'] +
                0.1 * score['completeness']
            )
            
            chunk['relevance_score'] = final_score
            scored.append(chunk)
        
        return sorted(scored, key=lambda x: x['relevance_score'], reverse=True)
```

#### Dynamic Context Sizing

```python
class DynamicContextSizer:
    def __init__(self):
        self.token_budgets = {
            'simple_query': 4000,
            'complex_query': 8000,
            'architectural_query': 12000,
            'debugging_query': 6000
        }
    
    def determine_context_size(self, query: str, query_type: str) -> int:
        """
        Dynamically determine optimal context size
        """
        base_budget = self.token_budgets.get(query_type, 6000)
        
        # Adjust based on query complexity
        complexity_factor = self._analyze_query_complexity(query)
        adjusted_budget = int(base_budget * complexity_factor)
        
        return min(adjusted_budget, 16000)  # Hard cap
    
    def _analyze_query_complexity(self, query: str) -> float:
        """
        Analyze query complexity to adjust context needs
        """
        complexity_indicators = {
            'architectural': ['architecture', 'design', 'pattern', 'structure'],
            'debugging': ['error', 'bug', 'issue', 'problem', 'debug'],
            'implementation': ['implement', 'create', 'build', 'develop'],
            'analysis': ['analyze', 'understand', 'explain', 'how']
        }
        
        complexity_score = 1.0
        for category, indicators in complexity_indicators.items():
            if any(indicator in query.lower() for indicator in indicators):
                if category == 'architectural':
                    complexity_score = max(complexity_score, 1.5)
                elif category == 'debugging':
                    complexity_score = max(complexity_score, 1.2)
                elif category == 'implementation':
                    complexity_score = max(complexity_score, 1.3)
        
        return min(complexity_score, 2.0)  # Cap at 2x
```

### 5. Metadata and Graph Enhancement

#### Code Complexity Metrics

```python
import ast
from typing import Dict, Any

class ComplexityAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def analyze_chunk(self, chunk: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate various complexity metrics for a code chunk
        """
        content = chunk['content']
        chunk_type = chunk.get('type', 'unknown')
        
        if chunk_type in ['function', 'method']:
            return self._analyze_function_complexity(content)
        elif chunk_type == 'class':
            return self._analyze_class_complexity(content)
        else:
            return self._analyze_general_complexity(content)
    
    def _analyze_function_complexity(self, code: str) -> Dict[str, float]:
        """
        Analyze function-specific complexity metrics
        """
        try:
            tree = ast.parse(code)
            function_node = tree.body[0]  # Assuming single function
            
            metrics = {
                'cyclomatic_complexity': self._cyclomatic_complexity(function_node),
                'nesting_depth': self._max_nesting_depth(function_node),
                'parameter_count': len(function_node.args.args),
                'line_count': len(code.split('\n')),
                'cognitive_complexity': self._cognitive_complexity(function_node)
            }
            
            return metrics
        except:
            return {'complexity_score': 1.0}  # Fallback
    
    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate McCabe cyclomatic complexity
        """
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ExceptHandler)):
                complexity += 1
        
        return complexity
```

#### Dependency Graph Construction

```python
import networkx as nx
from typing import Set, Dict, List

class DependencyGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.import_patterns = {
            'python': r'(?:from\s+(\S+)\s+import|import\s+(\S+))',
            'javascript': r'(?:import.*from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            'typescript': r'(?:import.*from\s+[\'"]([^\'"]+)[\'"]|import\s+[\'"]([^\'"]+)[\'"])',
            'java': r'import\s+([\w.]+)',
            'csharp': r'using\s+([\w.]+)'
        }
    
    def build_graph(self, chunks: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build dependency graph from code chunks
        """
        # Add nodes for each chunk
        for chunk in chunks:
            self.graph.add_node(
                chunk['id'], 
                **chunk,
                imports=self._extract_imports(chunk),
                exports=self._extract_exports(chunk)
            )
        
        # Add edges based on dependencies
        self._add_dependency_edges(chunks)
        
        return self.graph
    
    def _extract_imports(self, chunk: Dict[str, Any]) -> Set[str]:
        """
        Extract import statements from code chunk
        """
        content = chunk['content']
        language = chunk.get('language', 'unknown')
        
        if language not in self.import_patterns:
            return set()
        
        import re
        pattern = self.import_patterns[language]
        matches = re.findall(pattern, content)
        
        imports = set()
        for match in matches:
            if isinstance(match, tuple):
                imports.update([m for m in match if m])
            else:
                imports.add(match)
        
        return imports
    
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> List[str]:
        """
        Get chunks related through dependency relationships
        """
        if chunk_id not in self.graph:
            return []
        
        related = set()
        
        # Get dependencies (what this chunk imports)
        dependencies = list(self.graph.successors(chunk_id))
        related.update(dependencies)
        
        # Get dependents (what imports this chunk)
        dependents = list(self.graph.predecessors(chunk_id))
        related.update(dependents)
        
        # Expand to second-level relationships if needed
        if max_depth > 1:
            for dep in dependencies + dependents:
                second_level = list(self.graph.successors(dep)) + list(self.graph.predecessors(dep))
                related.update(second_level)
        
        related.discard(chunk_id)  # Remove self
        return list(related)
```

---

## Implementation Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Enhanced RAG System                         │
├─────────────────────────────────────────────────────────────────┤
│  Query Interface                                                │
│  ├── Query Expansion                                            │
│  ├── Intent Classification                                      │
│  └── Context Size Determination                                 │
├─────────────────────────────────────────────────────────────────┤
│  Retrieval Engine                                               │
│  ├── Hybrid Search (Semantic + Keyword)                        │
│  ├── Multi-level Ranking                                       │
│  └── Relevance Scoring                                         │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Base                                                 │
│  ├── Semantic Chunks (AST-based)                               │
│  ├── Vector Index (Code-specific embeddings)                   │
│  ├── Keyword Index (BM25)                                      │
│  ├── Dependency Graph                                          │
│  └── Metadata Store                                            │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion Pipeline                                             │
│  ├── Language Detection                                        │
│  ├── AST-based Chunking                                        │
│  ├── Complexity Analysis                                       │
│  ├── Embedding Generation                                      │
│  └── Index Building                                            │
├─────────────────────────────────────────────────────────────────┤
│  Video Memory Integration                                       │
│  ├── QR Code Generation                                        │
│  ├── Compressed Knowledge Storage                              │
│  └── Memory Retrieval                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components Design

#### 1. Enhanced Ingestion Pipeline

```python
class EnhancedIngestionPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.language_detector = LanguageDetector()
        self.chunkers = {
            'python': PythonChunker(),
            'javascript': JavaScriptChunker(),
            'typescript': TypeScriptChunker(),
            'java': JavaChunker(),
            'csharp': CSharpChunker(),
            'sql': SQLChunker(),
            'generic': GenericChunker()
        }
        self.embedding_manager = CodeEmbeddingManager()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_builder = DependencyGraphBuilder()
        
    async def process_codebase(self, codebase_path: str) -> Dict[str, Any]:
        """
        Process entire codebase with enhanced pipeline
        """
        processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Walk through codebase
        all_chunks = []
        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if self._should_process_file(file):
                    file_path = os.path.join(root, file)
                    chunks = await self._process_file(file_path)
                    all_chunks.extend(chunks)
                    processing_stats['processed_files'] += 1
                
                processing_stats['total_files'] += 1
        
        # Build indices and graphs
        indices = await self._build_indices(all_chunks)
        dependency_graph = self.dependency_builder.build_graph(all_chunks)
        
        processing_stats['total_chunks'] = len(all_chunks)
        processing_stats['processing_time'] = time.time() - start_time
        
        return {
            'chunks': all_chunks,
            'indices': indices,
            'dependency_graph': dependency_graph,
            'stats': processing_stats
        }
```

#### 2. Multi-Index Storage System

```python
class MultiIndexStorage:
    def __init__(self, storage_config: Dict[str, Any]):
        self.vector_store = self._init_vector_store(storage_config)
        self.keyword_store = self._init_keyword_store(storage_config)
        self.metadata_store = self._init_metadata_store(storage_config)
        self.graph_store = self._init_graph_store(storage_config)
        
    def store_chunk(self, chunk: Dict[str, Any], embedding: np.ndarray):
        """
        Store chunk across all indices
        """
        chunk_id = chunk['id']
        
        # Vector storage
        self.vector_store.add_vector(chunk_id, embedding)
        
        # Keyword storage
        self.keyword_store.add_document(chunk_id, chunk['content'])
        
        # Metadata storage
        self.metadata_store.store_metadata(chunk_id, chunk)
        
        # Graph storage (if has dependencies)
        if 'dependencies' in chunk:
            self.graph_store.add_node(chunk_id, chunk['dependencies'])
    
    def query_all_indices(self, query: str, query_embedding: np.ndarray, 
                         top_k: int = 10) -> Dict[str, List[str]]:
        """
        Query all indices and return ranked results
        """
        # Vector search
        vector_results = self.vector_store.similarity_search(
            query_embedding, top_k=top_k*2
        )
        
        # Keyword search  
        keyword_results = self.keyword_store.search(query, top_k=top_k*2)
        
        # Graph-based expansion
        graph_results = self.graph_store.expand_query_context(
            vector_results[:top_k//2]
        )
        
        return {
            'vector': vector_results,
            'keyword': keyword_results,
            'graph': graph_results
        }
```

### Integration with Existing Video Memory System

```python
class EnhancedVideoMemory:
    def __init__(self, base_video_memory):
        self.base_system = base_video_memory
        self.knowledge_compressor = KnowledgeCompressor()
        
    def generate_enhanced_video_memory(self, codebase_analysis: Dict[str, Any]) -> str:
        """
        Generate enhanced video memory with improved compression
        """
        # Extract key insights from enhanced analysis
        key_insights = self._extract_key_insights(codebase_analysis)
        
        # Compress using multiple strategies
        compressed_knowledge = self.knowledge_compressor.compress_multi_strategy(
            insights=key_insights,
            dependency_graph=codebase_analysis['dependency_graph'],
            complexity_metrics=codebase_analysis.get('complexity_metrics', {}),
            architectural_patterns=codebase_analysis.get('patterns', [])
        )
        
        # Generate QR code with enhanced payload
        qr_payload = {
            'version': '2.0',
            'compressed_knowledge': compressed_knowledge,
            'retrieval_hints': self._generate_retrieval_hints(codebase_analysis),
            'complexity_summary': self._generate_complexity_summary(codebase_analysis)
        }
        
        return self.base_system.generate_qr_code(qr_payload)
```

---

## Technical Specifications

### Required Dependencies

```json
{
  "dependencies": {
    "sentence-transformers": "^2.2.2",
    "transformers": "^4.25.1",
    "torch": "^1.13.1",
    "tree-sitter": "^0.20.1",
    "tree-sitter-python": "^0.20.2",
    "tree-sitter-javascript": "^0.20.0",
    "tree-sitter-typescript": "^0.20.2",
    "networkx": "^3.0",
    "scikit-learn": "^1.2.0",
    "rank-bm25": "^0.2.2",
    "faiss-cpu": "^1.7.3",
    "numpy": "^1.24.0",
    "pandas": "^1.5.2",
    "asyncio": "^3.4.3",
    "pydantic": "^1.10.4"
  },
  "optional_dependencies": {
    "openai": "^0.26.4",
    "chromadb": "^0.3.18",
    "redis": "^4.5.1",
    "elasticsearch": "^8.6.0"
  }
}
```

### Database Schema Extensions

```sql
-- Enhanced chunk metadata table
CREATE TABLE enhanced_chunks (
    id VARCHAR(255) PRIMARY KEY,
    file_path TEXT NOT NULL,
    chunk_type VARCHAR(50),
    start_line INTEGER,
    end_line INTEGER,
    function_name VARCHAR(255),
    class_name VARCHAR(255),
    complexity_score FLOAT,
    cyclomatic_complexity INTEGER,
    nesting_depth INTEGER,
    parameter_count INTEGER,
    line_count INTEGER,
    language VARCHAR(50),
    embedding_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_chunk_type (chunk_type),
    INDEX idx_complexity (complexity_score),
    INDEX idx_language (language),
    INDEX idx_file_path (file_path(255))
);

-- Dependency relationships table
CREATE TABLE chunk_dependencies (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    source_chunk_id VARCHAR(255),
    target_chunk_id VARCHAR(255),
    dependency_type VARCHAR(50),
    import_statement TEXT,
    strength FLOAT DEFAULT 1.0,
    
    FOREIGN KEY (source_chunk_id) REFERENCES enhanced_chunks(id),
    FOREIGN KEY (target_chunk_id) REFERENCES enhanced_chunks(id),
    INDEX idx_source (source_chunk_id),
    INDEX idx_target (target_chunk_id),
    INDEX idx_type (dependency_type)
);

-- Query performance metrics
CREATE TABLE query_performance (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    query_text TEXT,
    query_type VARCHAR(50),
    retrieval_method VARCHAR(50),
    response_time_ms INTEGER,
    relevance_score FLOAT,
    user_feedback INTEGER, -- 1-5 rating
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_query_type (query_type),
    INDEX idx_method (retrieval_method),
    INDEX idx_timestamp (timestamp)
);
```

### API Design

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Enhanced RAG Codebase Expert API", version="2.0.0")

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10
    include_context: bool = True
    retrieval_strategy: str = "hybrid"  # semantic, keyword, hybrid
    context_window_size: Optional[int] = None

class ChunkResponse(BaseModel):
    id: str
    content: str
    file_path: str
    chunk_type: str
    relevance_score: float
    complexity_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[ChunkResponse]
    total_found: int
    query_analysis: Dict[str, Any]
    processing_time_ms: float
    context_summary: Optional[str] = None

@app.post("/search", response_model=SearchResponse)
async def enhanced_search(request: QueryRequest) -> SearchResponse:
    """
    Enhanced search endpoint with hybrid retrieval
    """
    start_time = time.time()
    
    # Query analysis and expansion
    query_analyzer = QueryAnalyzer()
    analyzed_query = query_analyzer.analyze(request.query)
    
    # Determine optimal retrieval strategy
    if request.retrieval_strategy == "auto":
        strategy = query_analyzer.recommend_strategy(analyzed_query)
    else:
        strategy = request.retrieval_strategy
    
    # Execute search
    retriever = HybridRetriever(strategy=strategy)
    raw_results = await retriever.search(
        query=request.query,
        analyzed_query=analyzed_query,
        max_results=request.max_results
    )
    
    # Enhance results with context
    if request.include_context:
        context_assembler = ContextAssembler(
            max_tokens=request.context_window_size
        )
        context_summary = context_assembler.assemble_context(
            request.query, raw_results
        )
    else:
        context_summary = None
    
    # Format response
    results = [ChunkResponse(**result) for result in raw_results]
    
    return SearchResponse(
        results=results,
        total_found=len(results),
        query_analysis=analyzed_query,
        processing_time_ms=(time.time() - start_time) * 1000,
        context_summary=context_summary
    )

@app.post("/reindex")
async def trigger_reindexing(background_tasks: BackgroundTasks):
    """
    Trigger background reindexing with enhanced pipeline
    """
    background_tasks.add_task(run_enhanced_indexing)
    return {"message": "Reindexing started in background"}

@app.get("/stats")
async def get_system_stats():
    """
    Get current system statistics and health metrics
    """
    stats_collector = SystemStatsCollector()
    return await stats_collector.collect_comprehensive_stats()
```

### Configuration Management

```yaml
# config/enhanced_rag.yaml
rag_system:
  version: "2.0.0"
  
  embedding:
    primary_model: "microsoft/codebert-base"
    fallback_model: "all-MiniLM-L6-v2"
    batch_size: 32
    max_sequence_length: 512
    
  chunking:
    strategy: "ast_based"
    overlap_tokens: 50
    min_chunk_size: 50
    max_chunk_size: 2000
    preserve_functions: true
    preserve_classes: true
    
  retrieval:
    default_strategy: "hybrid"
    semantic_weight: 0.7
    keyword_weight: 0.3
    max_results: 20
    relevance_threshold: 0.6
    
  indexing:
    vector_index_type: "faiss"  # faiss, chromadb, pinecone
    keyword_index_type: "bm25"
    update_frequency: "incremental"  # full, incremental
    batch_processing: true
    
  performance:
    max_context_tokens: 8000
    query_timeout_seconds: 30
    cache_embeddings: true
    cache_ttl_hours: 24
    
  quality:
    enable_feedback_loop: true
    min_relevance_score: 0.5
    enable_query_analytics: true
    performance_monitoring: true

# Language-specific configurations
languages:
  python:
    chunker: "PythonASTChunker"
    complexity_analyzer: "PythonComplexityAnalyzer"
    embedding_strategy: "code_focused"
    
  javascript:
    chunker: "JavaScriptASTChunker"
    complexity_analyzer: "JSComplexityAnalyzer"
    embedding_strategy: "balanced"
    
  typescript:
    chunker: "TypeScriptASTChunker"
    complexity_analyzer: "TSComplexityAnalyzer"
    embedding_strategy: "type_aware"
```

---

## Implementation Phases

### Phase 1: Foundation Enhancement (Weeks 1-4)

**Objectives:**
- Implement AST-based chunking for major languages
- Integrate code-specific embedding models
- Basic hybrid search implementation

**Deliverables:**
```python
# Week 1-2: AST Chunking Implementation
class ASTChunkingSystem:
    def __init__(self):
        self.language_parsers = self._initialize_parsers()
        
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        language = self._detect_language(file_path)
        parser = self.language_parsers[language]
        return parser.chunk_file(file_path)

# Week 3-4: Embedding Model Integration
class CodeBERTEmbedding:
    def __init__(self):
        self.model = AutoModel.from_pretrained('microsoft/codebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
    def encode_chunks(self, chunks: List[str]) -> np.ndarray:
        return self._batch_encode(chunks)
```

**Success Metrics:**
- AST chunking implemented for Python, JavaScript, TypeScript
- CodeBERT integration functional
- 25% improvement in code search relevance
- Processing time under 2x current baseline

### Phase 2: Hybrid Retrieval System (Weeks 5-8)

**Objectives:**
- Implement BM25 keyword indexing
- Develop score fusion algorithms
- Create query expansion system

**Deliverables:**
```python
# Week 5-6: BM25 Integration
class BM25KeywordIndex:
    def __init__(self, corpus: List[str]):
        self.index = BM25Okapi(self._preprocess_corpus(corpus))
        
    def search(self, query: str, top_k: int) -> List[float]:
        return self.index.get_scores(self._preprocess_query(query))

# Week 7-8: Hybrid Search Engine
class HybridSearchEngine:
    def __init__(self, semantic_index, keyword_index):
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._fuse_results(
            self.semantic_index.search(query),
            self.keyword_index.search(query)
        )
```

**Success Metrics:**
- Hybrid search functional with configurable weights
- Query expansion improving recall by 30%
- Response time under 500ms for typical queries

### Phase 3: Advanced Context and Dependencies (Weeks 9-12)

**Objectives:**
- Implement dependency graph construction
- Develop intelligent context assembly
- Create complexity-aware ranking

**Deliverables:**
```python
# Week 9-10: Dependency Graph
class CodeDependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            self._extract_dependencies(chunk)
            self._add_to_graph(chunk)

# Week 11-12: Context Assembly
class IntelligentContextAssembler:
    def __init__(self, dependency_graph):
        self.graph = dependency_graph
        
    def assemble_context(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        return self._build_optimal_context(query, chunks)
```

**Success Metrics:**
- Dependency relationships correctly identified (>90% accuracy)
- Context assembly reduces irrelevant content by 40%
- Complex queries show 50% improvement in answer quality

### Phase 4: Performance and Production Optimization (Weeks 13-16)

**Objectives:**
- Implement caching and optimization
- Add monitoring and analytics
- Production deployment preparation

**Deliverables:**
```python
# Week 13-14: Performance Optimization
class CacheManager:
    def __init__(self):
        self.embedding_cache = TTLCache(maxsize=10000, ttl=3600)
        self.query_cache = TTLCache(maxsize=1000, ttl=1800)
        
    def get_cached_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        return self.embedding_cache.get(content_hash)

# Week 15-16: Monitoring and Analytics
class RAGAnalytics:
    def __init__(self):
        self.query_logger = QueryLogger()
        self.performance_monitor = PerformanceMonitor()
        
    def track_query_performance(self, query: str, results: List[Dict], 
                              response_time: float):
        self.query_logger.log_query(query, results, response_time)
        self.performance_monitor.record_metrics(response_time, len(results))
```

**Success Metrics:**
- System handles 100+ concurrent queries
- 95th percentile response time under 1 second
- Memory usage optimized (under 4GB for typical codebase)
- Comprehensive monitoring dashboard functional

---

## Performance and Scalability

### Performance Benchmarks

**Current System Baseline:**
- Query response time: 800ms (average)
- Memory usage: 2GB (typical codebase)
- Relevance score: 0.65 (user feedback)
- Index build time: 45 minutes (100k LOC)

**Enhanced System Targets:**
- Query response time: 300ms (average)
- Memory usage: 3.5GB (with enhanced features)
- Relevance score: 0.85 (user feedback)
- Index build time: 25 minutes (100k LOC)

### Scalability Architecture

```python
class ScalableRAGSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.load_balancer = self._init_load_balancer()
        self.index_shards = self._init_index_shards()
        self.cache_cluster = self._init_cache_cluster()
        
    def distribute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Distribute query across multiple index shards
        """
        # Determine optimal shards for query
        relevant_shards = self._select_shards(query)
        
        # Parallel query execution
        futures = []
        for shard in relevant_shards:
            future = self._query_shard_async(shard, query)
            futures.append(future)
        
        # Collect and merge results
        all_results = []
        for future in futures:
            shard_results = await future
            all_results.extend(shard_results)
        
        # Global ranking and filtering
        return self._merge_and_rank_results(all_results)
    
    def _select_shards(self, query: str) -> List[str]:
        """
        Intelligently select relevant shards based on query
        """
        # Analyze query to determine relevant code areas
        query_analyzer = QueryAnalyzer()
        query_features = query_analyzer.extract_features(query)
        
        # Map to relevant shards
        shard_scores = {}
        for shard_id, shard_metadata in self.index_shards.items():
            similarity = self._calculate_shard_similarity(
                query_features, shard_metadata
            )
            if similarity > 0.3:  # Threshold for relevance
                shard_scores[shard_id] = similarity
        
        # Return top shards
        return sorted(shard_scores.keys(), 
                     key=lambda x: shard_scores[x], 
                     reverse=True)[:3]
```

### Memory Optimization Strategies

```python
class MemoryOptimizer:
    def __init__(self):
        self.compression_strategies = {
            'embeddings': EmbeddingCompressor(),
            'indices': IndexCompressor(),
            'metadata': MetadataCompressor()
        }
        
    def optimize_memory_usage(self, system_components: Dict[str, Any]):
        """
        Apply memory optimization across system components
        """
        optimizations = {}
        
        # Compress embeddings using quantization
        if 'embeddings' in system_components:
            compressed_embeddings = self.compression_strategies['embeddings'].compress(
                system_components['embeddings']
            )
            optimizations['embeddings'] = {
                'original_size': self._calculate_size(system_components['embeddings']),
                'compressed_size': self._calculate_size(compressed_embeddings),
                'compression_ratio': self._calculate_compression_ratio(
                    system_components['embeddings'], compressed_embeddings
                )
            }
        
        # Optimize index structures
        if 'indices' in system_components:
            optimized_indices = self.compression_strategies['indices'].optimize(
                system_components['indices']
            )
            optimizations['indices'] = {
                'optimization_applied': True,
                'memory_saved': self._calculate_memory_saved(
                    system_components['indices'], optimized_indices
                )
            }
        
        return optimizations

class EmbeddingCompressor:
    def __init__(self):
        self.quantization_bits = 8  # 8-bit quantization
        
    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply quantization to reduce embedding memory footprint
        """
        # Normalize embeddings to [0, 1] range
        normalized = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
        
        # Quantize to 8-bit integers
        quantized = (normalized * 255).astype(np.uint8)
        
        return quantized
    
    def decompress(self, compressed_embeddings: np.ndarray, 
                  original_min: float, original_max: float) -> np.ndarray:
        """
        Decompress quantized embeddings for use
        """
        # Convert back to float and denormalize
        normalized = compressed_embeddings.astype(np.float32) / 255.0
        decompressed = normalized * (original_max - original_min) + original_min
        
        return decompressed
```

---

## Testing and Validation

### Unit Testing Strategy

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestEnhancedRAGSystem:
    
    @pytest.fixture
    def sample_code_chunks(self):
        return [
            {
                'id': 'chunk_1',
                'content': 'def calculate_score(data):\n    return sum(data) / len(data)',
                'type': 'function',
                'language': 'python',
                'complexity_score': 1.2
            },
            {
                'id': 'chunk_2', 
                'content': 'class DataProcessor:\n    def __init__(self):\n        self.data = []',
                'type': 'class',
                'language': 'python',
                'complexity_score': 2.1
            }
        ]
    
    @pytest.fixture
    def mock_embedding_model(self):
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        return mock_model
    
    def test_ast_chunking_preserves_function_boundaries(self):
        """Test that AST chunking correctly identifies function boundaries"""
        chunker = PythonASTChunker()
        code = """
def function_one():
    return "first"

def function_two():
    return "second"
        """
        
        chunks = chunker.chunk_code(code)
        
        assert len(chunks) == 2
        assert chunks[0]['type'] == 'function'
        assert 'function_one' in chunks[0]['content']
        assert 'function_two' in chunks[1]['content']
    
    def test_hybrid_search_combines_results(self, sample_code_chunks, mock_embedding_model):
        """Test that hybrid search properly combines semantic and keyword results"""
        # Setup
        hybrid_retriever = HybridRetriever(
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        with patch.object(hybrid_retriever, 'semantic_search') as mock_semantic, \
             patch.object(hybrid_retriever, 'keyword_search') as mock_keyword:
            
            mock_semantic.return_value = [
                {'chunk_id': 'chunk_1', 'score': 0.9},
                {'chunk_id': 'chunk_2', 'score': 0.7}
            ]
            mock_keyword.return_value = [
                {'chunk_id': 'chunk_2', 'score': 0.8},
                {'chunk_id': 'chunk_1', 'score': 0.6}
            ]
            
            # Execute
            results = hybrid_retriever.search("calculate score", top_k=2)
            
            # Verify
            assert len(results) == 2
            assert results[0]['chunk_id'] == 'chunk_1'  # Higher combined score
            assert 'combined_score' in results[0]
    
    def test_complexity_analysis_calculates_metrics(self):
        """Test that complexity analysis produces expected metrics"""
        analyzer = ComplexityAnalyzer()
        
        complex_function = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            for sub_item in item.values():
                if sub_item is not None:
                    result.append(sub_item * 2)
                else:
                    result.append(0)
        else:
            result.append(-1)
    return result
        """
        
        metrics = analyzer.analyze_function(complex_function)
        
        assert metrics['cyclomatic_complexity'] > 1
        assert metrics['nesting_depth'] >= 3
        assert metrics['line_count'] > 5
    
    def test_dependency_graph_construction(self):
        """Test dependency graph correctly identifies relationships"""
        chunks = [
            {
                'id': 'chunk_1',
                'content': 'import utils\nfrom math import sqrt',
                'imports': ['utils', 'math.sqrt']
            },
            {
                'id': 'chunk_2',
                'content': 'def utility_function():\n    pass',
                'exports': ['utility_function']
            }
        ]
        
        graph_builder = DependencyGraphBuilder()
        graph = graph_builder.build_graph(chunks)
        
        assert graph.number_of_nodes() == 2
        # Should have edge from chunk_1 to chunk_2 if utils refers to utility_function
        assert len(list(graph.predecessors('chunk_1'))) >= 0
```

### Integration Testing

```python
class TestRAGSystemIntegration:
    
    @pytest.fixture
    def full_rag_system(self):
        config = {
            'embedding_model': 'microsoft/codebert-base',
            'chunking_strategy': 'ast_based',
            'retrieval_strategy': 'hybrid'
        }
        return EnhancedRAGSystem(config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self, full_rag_system):
        """Test complete query processing pipeline"""
        # Setup test codebase
        test_codebase = create_test_codebase()
        
        # Index the codebase
        await full_rag_system.index_codebase(test_codebase)
        
        # Execute query
        query = "How to calculate user scores?"
        results = await full_rag_system.query(query)
        
        # Verify results
        assert len(results) > 0
        assert all('relevance_score' in result for result in results)
        assert results[0]['relevance_score'] > 0.5
    
    def test_performance_under_load(self, full_rag_system):
        """Test system performance under concurrent load"""
        import concurrent.futures
        import time
        
        queries = [
            "authentication implementation",
            "database connection setup", 
            "error handling patterns",
            "logging configuration",
            "API endpoint definitions"
        ]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(full_rag_system.query, query)
                for query in queries * 4  # 20 total queries
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 10.0  # Should complete in under 10 seconds
        assert all(len(result) > 0 for result in results)  # All queries return results
        assert total_time / len(results) < 0.5  # Average under 500ms per query
```

### Performance Testing

```python
class TestPerformanceMetrics:
    
    def test_memory_usage_within_limits(self):
        """Test that memory usage stays within acceptable limits"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Index a large codebase
        rag_system = EnhancedRAGSystem()
        rag_system.index_large_codebase(size="100k_loc")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 4GB additional memory
        assert memory_increase < 4000  # 4GB in MB
    
    def test_query_response_times(self):
        """Test query response times meet performance targets"""
        rag_system = EnhancedRAGSystem()
        
        query_types = {
            'simple': "find function",
            'complex': "explain authentication flow and error handling patterns",
            'architectural': "show me the overall system architecture and component relationships"
        }
        
        response_times = {}
        
        for query_type, query in query_types.items():
            start_time = time.time()
            results = rag_system.query(query)
            end_time = time.time()
            
            response_times[query_type] = (end_time - start_time) * 1000  # ms
        
        # Performance targets
        assert response_times['simple'] < 200  # 200ms for simple queries
        assert response_times['complex'] < 500  # 500ms for complex queries  
        assert response_times['architectural'] < 1000  # 1s for architectural queries
    
    def test_relevance_quality_metrics(self):
        """Test that search relevance meets quality standards"""
        rag_system = EnhancedRAGSystem()
        
        # Test queries with known expected results
        test_cases = [
            {
                'query': 'user authentication',
                'expected_files': ['auth.py', 'login.js', 'user_manager.py'],
                'min_relevance': 0.7
            },
            {
                'query': 'database connection',
                'expected_files': ['db_config.py', 'connection.js', 'models.py'],
                'min_relevance': 0.8
            }
        ]
        
        for test_case in test_cases:
            results = rag_system.query(test_case['query'])
            
            # Check that expected files appear in results
            result_files = [r['file_path'] for r in results[:5]]
            
            relevance_scores = [r['relevance_score'] for r in results[:3]]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            
            assert avg_relevance >= test_case['min_relevance']
            
            # At least one expected file should appear in top 5 results
            assert any(
                any(expected in result_file for expected in test_case['expected_files'])
                for result_file in result_files
            )
```

### Validation Framework

```python
class RAGValidationFramework:
    def __init__(self):
        self.validators = {
            'chunking': ChunkingValidator(),
            'embedding': EmbeddingValidator(), 
            'retrieval': RetrievalValidator(),
            'context': ContextValidator()
        }
        
    def validate_system_upgrade(self, old_system, new_system) -> Dict[str, Any]:
        """
        Comprehensive validation when upgrading RAG system
        """
        validation_results = {}
        
        # Test suite of queries with ground truth
        test_queries = self._load_test_queries()
        
        for validator_name, validator in self.validators.items():
            print(f"Running {validator_name} validation...")
            
            old_results = validator.evaluate_system(old_system, test_queries)
            new_results = validator.evaluate_system(new_system, test_queries)
            
            validation_results[validator_name] = {
                'old_performance': old_results,
                'new_performance': new_results,
                'improvement': validator.calculate_improvement(old_results, new_results),
                'regression_detected': validator.detect_regression(old_results, new_results)
            }
        
        return validation_results
    
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """
        Load standardized test queries with expected results
        """
        return [
            {
                'query': 'user authentication implementation',
                'expected_chunks': ['auth_chunk_1', 'auth_chunk_2'],
                'expected_relevance': 0.8,
                'query_type': 'implementation'
            },
            {
                'query': 'explain the database schema',
                'expected_chunks': ['schema_chunk_1', 'model_chunk_1'],
                'expected_relevance': 0.75,
                'query_type': 'explanation'
            },
            # ... more test cases
        ]

class RetrievalValidator:
    def evaluate_system(self, system, test_queries) -> Dict[str, float]:
        """
        Evaluate retrieval quality metrics
        """
        metrics = {
            'precision_at_5': 0.0,
            'recall_at_10': 0.0,
            'mrr': 0.0,  # Mean Reciprocal Rank
            'ndcg': 0.0   # Normalized Discounted Cumulative Gain
        }
        
        for query_data in test_queries:
            results = system.query(query_data['query'])
            
            # Calculate precision@5
            relevant_in_top5 = sum(
                1 for r in results[:5] 
                if r['chunk_id'] in query_data['expected_chunks']
            )
            metrics['precision_at_5'] += relevant_in_top5 / 5
            
            # Calculate recall@10  
            relevant_in_top10 = sum(
                1 for r in results[:10]
                if r['chunk_id'] in query_data['expected_chunks']
            )
            metrics['recall_at_10'] += relevant_in_top10 / len(query_data['expected_chunks'])
        
        # Average across all queries
        for metric in metrics:
            metrics[metric] /= len(test_queries)
            
        return metrics
```

---

## Conclusion

This comprehensive enhancement strategy provides a roadmap for significantly improving the RAG system's capabilities while maintaining the innovative video memory approach. The proposed improvements address key limitations in the current system and introduce advanced techniques from modern information retrieval and natural language processing.

**Key Expected Outcomes:**

1. **Improved Relevance**: 30-40% improvement in search relevance through code-specific embeddings and hybrid retrieval
2. **Better Code Understanding**: AST-based chunking preserves semantic boundaries and function relationships
3. **Intelligent Context**: Dynamic context assembly reduces noise and improves answer quality
4. **Scalable Architecture**: Distributed processing enables handling of larger codebases
5. **Enhanced User Experience**: Faster response times and more accurate results

**Implementation Priority:**
1. AST-based chunking (immediate impact, foundational)
2. Code-specific embeddings (significant relevance improvement)
3. Hybrid retrieval (improved recall and precision)
4. Intelligent context assembly (better answer quality)
5. Performance optimization (production readiness)

The modular design allows for incremental implementation and testing, ensuring each component can be validated independently before full system integration. This approach minimizes risk while maximizing the potential for significant improvements in code search and understanding capabilities.