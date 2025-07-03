# rag_components/context_assembler.py

from typing import List, Dict, Any, Set
import re
from collections import defaultdict

class ContextAssembler:
    """
    Intelligent context assembly system that prioritizes, deduplicates, 
    and orders retrieved chunks for optimal LLM consumption.
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def _extract_file_path(self, chunk: Dict[str, Any]) -> str:
        """Extract file path from chunk metadata."""
        metadata = chunk.get('metadata', {})
        return metadata.get('file_path', 'unknown')
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate basic content similarity using token overlap."""
        tokens1 = set(re.findall(r'\w+', content1.lower()))
        tokens2 = set(re.findall(r'\w+', content2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Remove highly similar chunks to avoid redundancy."""
        if not chunks:
            return chunks
        
        deduplicated = []
        seen_contents = []
        
        for chunk in chunks:
            content = chunk['content']
            is_duplicate = False
            
            for seen_content in seen_contents:
                if self._calculate_content_similarity(content, seen_content) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
                seen_contents.append(content)
        
        return deduplicated
    
    def _group_by_file(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their source file."""
        file_groups = defaultdict(list)
        
        for chunk in chunks:
            file_path = self._extract_file_path(chunk)
            file_groups[file_path].append(chunk)
        
        return dict(file_groups)
    
    def _order_chunks_within_file(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order chunks within a file by their line numbers if available."""
        def get_start_line(chunk):
            metadata = chunk.get('metadata', {})
            return metadata.get('start_line', 0)
        
        return sorted(chunks, key=get_start_line)
    
    def _calculate_file_priority(self, file_path: str, chunks: List[Dict[str, Any]]) -> float:
        """Calculate priority score for a file based on relevance and chunk quality."""
        if not chunks:
            return 0.0
        
        # Average relevance score
        avg_relevance = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # File type bonus (prioritize certain file types)
        file_type_bonus = 0.0
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.cs')):
            file_type_bonus = 0.1
        elif file_path.endswith(('.md', '.txt', '.rst')):
            file_type_bonus = 0.05
        
        # Penalize very common files
        file_name = file_path.lower()
        if any(common in file_name for common in ['test', 'spec', '__pycache__', 'node_modules']):
            file_type_bonus -= 0.1
        
        return avg_relevance + file_type_bonus
    
    def _format_chunk_for_display(self, chunk: Dict[str, Any], include_metadata: bool = True) -> str:
        """Format a chunk for display in the final context."""
        content = chunk['content']
        file_path = self._extract_file_path(chunk)
        
        if include_metadata:
            score = chunk.get('relevance_score', 0.0)
            metadata = chunk.get('metadata', {})
            start_line = metadata.get('start_line', '')
            
            header_parts = [f"File: {file_path}"]
            if start_line:
                header_parts.append(f"Line: {start_line}")
            header_parts.append(f"Score: {score:.3f}")
            
            header = " | ".join(header_parts)
            return f"```\n{header}\n{content}\n```"
        else:
            return f"```\n{content}\n```"
    
    def assemble_context(self, retrieved_chunks: List[Dict[str, Any]], 
                        max_chunks: int = 10,
                        max_chars: int = 8000,
                        deduplicate: bool = True,
                        group_by_files: bool = True,
                        include_metadata: bool = True) -> str:
        """
        Assemble retrieved chunks into an intelligent, well-formatted context.
        
        Args:
            retrieved_chunks: List of chunks from retrieval
            max_chunks: Maximum number of chunks to include
            max_chars: Maximum character limit for the context
            deduplicate: Whether to remove similar chunks
            group_by_files: Whether to group chunks by source file
            include_metadata: Whether to include metadata in output
        """
        if not retrieved_chunks:
            return "No relevant code found for your query."
        
        # 1. Deduplicate if requested
        chunks = self._deduplicate_chunks(retrieved_chunks) if deduplicate else retrieved_chunks
        
        # 2. Limit to max_chunks
        chunks = chunks[:max_chunks]
        
        if group_by_files:
            # 3. Group by files and order
            file_groups = self._group_by_file(chunks)
            
            # Calculate file priorities
            file_priorities = {}
            for file_path, file_chunks in file_groups.items():
                file_priorities[file_path] = self._calculate_file_priority(file_path, file_chunks)
            
            # Sort files by priority
            sorted_files = sorted(file_priorities.keys(), 
                                key=lambda f: file_priorities[f], reverse=True)
            
            # 4. Assemble context
            context_parts = []
            current_chars = 0
            
            for file_path in sorted_files:
                file_chunks = self._order_chunks_within_file(file_groups[file_path])
                
                if len(file_chunks) > 1:
                    file_header = f"\n## {file_path} ({len(file_chunks)} chunks)\n"
                else:
                    file_header = f"\n## {file_path}\n"
                
                # Check if adding this file would exceed limit
                file_content = file_header
                chunk_contents = []
                
                for chunk in file_chunks:
                    chunk_text = self._format_chunk_for_display(chunk, include_metadata)
                    chunk_contents.append(chunk_text)
                
                file_content += "\n\n".join(chunk_contents)
                
                if current_chars + len(file_content) > max_chars and context_parts:
                    break
                
                context_parts.append(file_content)
                current_chars += len(file_content)
        
        else:
            # Simple linear assembly without grouping
            context_parts = []
            current_chars = 0
            
            for chunk in chunks:
                chunk_text = self._format_chunk_for_display(chunk, include_metadata)
                
                if current_chars + len(chunk_text) > max_chars and context_parts:
                    break
                
                context_parts.append(chunk_text)
                current_chars += len(chunk_text)
        
        # 5. Create final context
        if not context_parts:
            return "No relevant code found within the character limit."
        
        final_context = "".join(context_parts)
        
        # Add summary header
        num_chunks = len([p for p in context_parts if "```" in p])
        summary = f"# Search Results ({num_chunks} code chunks found)\n"
        
        return summary + final_context
    
    def get_expanded_context(self, chunk_id: str, context_lines: int = 5) -> str:
        """
        Get expanded context around a specific chunk by retrieving neighboring chunks.
        """
        chunk = self.kb.get_chunk_by_id(chunk_id)
        if not chunk:
            return "Chunk not found."
        
        file_path = self._extract_file_path(chunk)
        metadata = chunk.get('metadata', {})
        start_line = metadata.get('start_line', 0)
        
        # Find chunks from the same file
        same_file_chunks = []
        for other_chunk in self.kb.doc_store.values():
            if self._extract_file_path(other_chunk) == file_path:
                same_file_chunks.append(other_chunk)
        
        # Sort by line number and find neighbors
        same_file_chunks = self._order_chunks_within_file(same_file_chunks)
        
        # Find the target chunk and get neighbors
        target_index = -1
        for i, c in enumerate(same_file_chunks):
            if c['id'] == chunk_id:
                target_index = i
                break
        
        if target_index == -1:
            return self._format_chunk_for_display(chunk)
        
        # Get surrounding chunks
        start_idx = max(0, target_index - context_lines)
        end_idx = min(len(same_file_chunks), target_index + context_lines + 1)
        
        context_chunks = same_file_chunks[start_idx:end_idx]
        
        # Mark the target chunk
        formatted_chunks = []
        for c in context_chunks:
            is_target = c['id'] == chunk_id
            chunk_text = self._format_chunk_for_display(c, include_metadata=True)
            if is_target:
                chunk_text = f"**TARGET CHUNK**\n{chunk_text}"
            formatted_chunks.append(chunk_text)
        
        return f"# Expanded Context for {file_path}\n\n" + "\n\n".join(formatted_chunks)