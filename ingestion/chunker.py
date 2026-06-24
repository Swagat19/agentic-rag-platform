
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# Detect a markdown table separator row: '|---|---|...' (any number of
# dashes / colons / pipes / spaces). Used to anchor table-block detection
# so we keep the header row attached to each split.
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|?\s*$")


def _is_table_line(line: str) -> bool:
    """A line that belongs to a markdown table starts with '|' once stripped."""
    return line.lstrip().startswith("|")


def _find_table_blocks(lines: List[str], min_lines: int = 3) -> List[Tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for every contiguous run of
    >= ``min_lines`` markdown-table lines in ``lines``.

    A table block is any maximal run of consecutive lines where every line
    starts with '|'. We do not require a separator row because Docling
    sometimes emits separator-first / header-second tables.
    """
    blocks: List[Tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            j = i
            while j < n and _is_table_line(lines[j]):
                j += 1
            if j - i >= min_lines:
                blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


# How many leading lines of a markdown table are treated as the "header
# block" that gets repeated on every split. Three covers both standard
# markdown (header / separator / first-data) and Docling's title /
# separator / column-headers / data layout. Empirically validated against
# the FY2024 KPI table in sr2024.pdf which has the latter shape.
_TABLE_HEADER_ROWS = 3


def _split_table_block(
    table_lines: List[str],
    rows_per_chunk: int,
    max_chars_per_chunk: int = 10000,
) -> List[str]:
    """Split a single table block into pieces capped by **both** a row
    count and a character budget, repeating the first
    ``_TABLE_HEADER_ROWS`` lines on every piece so each chunk carries
    the column-header context its embedding needs.

    Why two limits: the FY2024 KPI table has uniform ~1 k-char rows, so
    a row-count cap is enough. The TCFD risk-disclosure tables in the
    same PDF have *one* row that is a multi-paragraph risk description
    spanning 5-10 k chars; capping only on row count would still
    produce 30-43 k chunks for those tables. We always include at least
    one body row per chunk so a single oversized row still ends up in
    its own chunk rather than being dropped.
    """
    if len(table_lines) <= _TABLE_HEADER_ROWS:
        return ["\n".join(table_lines)]

    prefix = table_lines[:_TABLE_HEADER_ROWS]
    body = [r for r in table_lines[_TABLE_HEADER_ROWS:] if r.strip()]

    if not body:
        return ["\n".join(prefix)]

    # +1 per line for the newline that join() adds back.
    prefix_size = sum(len(p) + 1 for p in prefix)

    groups: List[List[str]] = []
    current: List[str] = []
    current_size = prefix_size
    for row in body:
        row_size = len(row) + 1
        full_by_rows = len(current) >= rows_per_chunk
        # Only honour the char cap once we already have at least one row;
        # otherwise an oversized row would be silently dropped.
        full_by_chars = bool(current) and (current_size + row_size > max_chars_per_chunk)
        if full_by_rows or full_by_chars:
            groups.append(current)
            current = []
            current_size = prefix_size
        current.append(row)
        current_size += row_size
    if current:
        groups.append(current)

    return ["\n".join(prefix + g) for g in groups]


def split_markdown_by_tables(
    text: str,
) -> List[Tuple[str, str]]:
    """Return ``[(segment_type, segment_content), ...]`` where
    ``segment_type`` is either 'prose' or 'table'. Prose segments keep
    their original markdown verbatim (including images, headings, lists).
    Table segments contain a single markdown table block.

    Tables shorter than 3 lines are treated as prose -- they are too small
    to merit special handling and would inflate the chunk count.
    """
    lines = text.split("\n")
    blocks = _find_table_blocks(lines)
    segments: List[Tuple[str, str]] = []
    cursor = 0
    for start, end in blocks:
        if start > cursor:
            segments.append(("prose", "\n".join(lines[cursor:start])))
        segments.append(("table", "\n".join(lines[start:end])))
        cursor = end
    if cursor < len(lines):
        segments.append(("prose", "\n".join(lines[cursor:])))
    return segments


@dataclass
class ChunkingConfig:
    """Simple configuration for PDF chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    use_semantic_splitting: bool = True

    # Table-aware chunking. When enabled, markdown tables are detected
    # before semantic chunking and split into row groups with the header
    # repeated on each group. Disable to fall back to whole-document
    # semantic / recursive splitting.
    use_table_aware_chunking: bool = True
    # 4 keeps each table chunk ~5-9 k chars on the FY2024 KPI table while
    # still preserving 3 leading header rows; tuned against the
    # ``test_chunker_emits_separate_chunks_per_table_group`` fixture.
    table_rows_per_chunk: int = 4
    # Hard ceiling on per-chunk size for table splits. Forces narrative
    # tables (TCFD-style multi-paragraph cells) to split even when the
    # row count is small. ~10 k chars fits comfortably inside an 8 k-token
    # embedding context window for either nomic-embed-text or
    # text-embedding-3-small.
    table_max_chars_per_chunk: int = 10000

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        if self.table_rows_per_chunk <= 0:
            raise ValueError("table_rows_per_chunk must be positive")
        
@dataclass
class DocumentChunk:
    """Represents a document chunk."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            self.token_count = len(self.content) // 4
            
class PDFSemanticChunker:
    """ Semantic chunker for PDF documents."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        )
        
        # Semantic splitter
        if config.use_semantic_splitting:
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile" # Split %10 diff.
            )
        
        # Recursive splitter
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def _split_prose(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Run the existing semantic / recursive splitter on a prose-only segment."""
        doc = Document(page_content=content, metadata=dict(base_metadata))
        try:
            if (
                self.config.use_semantic_splitting
                and len(content) > self.config.chunk_size
            ):
                chunks = self.semantic_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "semantic"
            else:
                chunks = self.fallback_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "recursive"
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            chunks = self.fallback_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["chunk_method"] = "fallback"
        return chunks

    def chunk_content(
        self,
        content: str,
        title: str = "PDF Document",
        source: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk PDF content into semantic pieces and convert to DocumentChunk objects.

        When ``use_table_aware_chunking`` is enabled (default), the markdown
        is first segmented by table boundaries. Prose segments flow through
        the existing semantic / recursive splitter; table segments are split
        row-by-row with the header repeated on every chunk. This prevents
        large KPI tables from collapsing into a single high-dimensional
        chunk whose embedding is too diluted for vector retrieval to rank.
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "content_type": "pdf",
            **(metadata or {})
        }

        chunks: List[Document] = []

        if self.config.use_table_aware_chunking:
            segments = split_markdown_by_tables(content)
            table_idx = 0
            for seg_type, seg_content in segments:
                if not seg_content.strip():
                    continue
                if seg_type == "table":
                    table_lines = seg_content.split("\n")
                    splits = _split_table_block(
                        table_lines,
                        self.config.table_rows_per_chunk,
                        self.config.table_max_chars_per_chunk,
                    )
                    for part_idx, part in enumerate(splits):
                        meta = dict(base_metadata)
                        meta.update(
                            {
                                "chunk_method": "table_aware",
                                "segment_type": "table",
                                "table_id": table_idx,
                                "table_part": part_idx,
                                "table_total_parts": len(splits),
                            }
                        )
                        chunks.append(Document(page_content=part, metadata=meta))
                    table_idx += 1
                else:
                    prose_chunks = self._split_prose(seg_content, base_metadata)
                    for chunk in prose_chunks:
                        chunk.metadata.setdefault("segment_type", "prose")
                    chunks.extend(prose_chunks)
        else:
            chunks = self._split_prose(content, base_metadata)

        # Filter small chunks and convert to DocumentChunk
        final_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if len(text) >= self.config.min_chunk_size:
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(text)
                })
                final_chunks.append(
                    DocumentChunk(
                        content=text,
                        index=i,
                        start_char=0,
                        end_char=len(text),
                        metadata=chunk.metadata
                    )
                )

        return final_chunks

    def chunk_pdf_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk PDF documents.
        
        Args:
            documents: List of PDF Document objects
            
        Returns:
            List of chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            title = doc.metadata.get("title", "PDF Document")
            source = doc.metadata.get("source", "pdf")
            
            chunks = self.chunk_content(
                content=doc.page_content,
                title=title,
                source=source,
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)
        
        return all_chunks


def create_chunker(config: ChunkingConfig) -> PDFSemanticChunker:
    """Create PDF chunker with simple configuration."""
    return PDFSemanticChunker(config)
