-- Migration: add kpi_facts table + search_kpi_facts function.
-- Idempotent: safe to re-run. Used for in-place upgrade of an already
-- initialised DB; the canonical definitions live in sql/schema.sql.

CREATE TABLE IF NOT EXISTS kpi_facts (
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

CREATE INDEX IF NOT EXISTS idx_kpi_facts_metric_trgm ON kpi_facts USING GIN (metric_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_kpi_facts_metric_tsv ON kpi_facts USING GIN (to_tsvector('english', metric_name));
CREATE INDEX IF NOT EXISTS idx_kpi_facts_category ON kpi_facts (category);
CREATE INDEX IF NOT EXISTS idx_kpi_facts_year ON kpi_facts (year);
CREATE INDEX IF NOT EXISTS idx_kpi_facts_source_chunk ON kpi_facts (source_chunk_id);

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
