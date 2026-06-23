CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS kpi_facts CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_chunks_document_id;
DROP INDEX IF EXISTS idx_documents_metadata;
DROP INDEX IF EXISTS idx_chunks_content_trgm;
DROP INDEX IF EXISTS idx_kpi_facts_metric_trgm;
DROP INDEX IF EXISTS idx_kpi_facts_metric_tsv;
DROP INDEX IF EXISTS idx_kpi_facts_category;
DROP INDEX IF EXISTS idx_kpi_facts_year;
DROP INDEX IF EXISTS idx_kpi_facts_source_chunk;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);

-- Embedding dimension is 768 to match Ollama's `nomic-embed-text` model
-- (the default in .env.example). If you switch to OpenAI's
-- `text-embedding-3-small` (1536 dim) or a different model, update this
-- column and both function signatures below, then reinitialise the DB.
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks (document_id, chunk_index);
CREATE INDEX idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);

-- Structured fact table populated by `scripts/extract_kpi_facts.py`.
-- Each row is a (metric, value, year, scope, category) tuple extracted by an
-- LLM pass over the chunks. Source-traced via source_chunk_id so the agent's
-- SQL tool can return both the structured fact and the supporting chunk.
-- Extraction is intentionally noisy: the eval framework measures whether
-- agents querying this table outperform pure vector/keyword retrieval on
-- lookup-style questions, not whether the table is a perfect ground truth.
CREATE TABLE kpi_facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT,
    year INTEGER,
    scope TEXT,
    baseline_year INTEGER,
    category TEXT,
    source_chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    extracted_text TEXT,
    confidence REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kpi_facts_metric_trgm ON kpi_facts USING GIN (metric_name gin_trgm_ops);
CREATE INDEX idx_kpi_facts_metric_tsv ON kpi_facts USING GIN (to_tsvector('english', metric_name));
CREATE INDEX idx_kpi_facts_category ON kpi_facts (category);
CREATE INDEX idx_kpi_facts_year ON kpi_facts (year);
CREATE INDEX idx_kpi_facts_source_chunk ON kpi_facts (source_chunk_id);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_sessions_user_id ON sessions (user_id);
CREATE INDEX idx_sessions_expires_at ON sessions (expires_at);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_session_id ON messages (session_id, created_at);


CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        (1 - (c.embedding <=> query_embedding))::double precision AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(768),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            (1 - (c.embedding <=> query_embedding))::double precision AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::double precision AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (
            COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
            COALESCE(t.text_sim, 0)::double precision * text_weight
        ) AS combined_score,
        COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
        COALESCE(t.text_sim, 0)::double precision AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id AS chunk_id,
        chunks.content,
        chunks.chunk_index,
        chunks.metadata
    FROM chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
END;
$$;

-- Lookup function for the agent's sql_kpi_search tool.
-- Combines trigram similarity on metric_name with optional year/category
-- filters. Returns the structured fact plus the supporting chunk content
-- so the agent can both quote the number and cite the surrounding text.
-- A passing similarity floor of 0.10 keeps obvious junk out; tune via the
-- min_similarity arg if needed.
CREATE OR REPLACE FUNCTION search_kpi_facts(
    query_text TEXT,
    target_year INT DEFAULT NULL,
    target_category TEXT DEFAULT NULL,
    match_count INT DEFAULT 10,
    min_similarity REAL DEFAULT 0.10
)
RETURNS TABLE (
    fact_id UUID,
    metric_name TEXT,
    value TEXT,
    unit TEXT,
    year INTEGER,
    scope TEXT,
    baseline_year INTEGER,
    category TEXT,
    similarity REAL,
    source_chunk_id UUID,
    source_document_id UUID,
    chunk_content TEXT,
    document_title TEXT,
    document_source TEXT,
    extracted_text TEXT,
    confidence REAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kf.id AS fact_id,
        kf.metric_name,
        kf.value,
        kf.unit,
        kf.year,
        kf.scope,
        kf.baseline_year,
        kf.category,
        similarity(kf.metric_name, query_text) AS similarity,
        kf.source_chunk_id,
        kf.source_document_id,
        c.content AS chunk_content,
        d.title AS document_title,
        d.source AS document_source,
        kf.extracted_text,
        kf.confidence
    FROM kpi_facts kf
    JOIN chunks c ON kf.source_chunk_id = c.id
    JOIN documents d ON kf.source_document_id = d.id
    WHERE similarity(kf.metric_name, query_text) >= min_similarity
      AND (target_year IS NULL OR kf.year IS NULL OR kf.year = target_year)
      AND (target_category IS NULL OR kf.category = target_category)
    ORDER BY similarity DESC, kf.confidence DESC NULLS LAST
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    AVG(c.token_count) AS avg_tokens_per_chunk,
    SUM(c.token_count) AS total_tokens
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata;
