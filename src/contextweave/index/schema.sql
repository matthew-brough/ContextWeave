CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);

CREATE TABLE files (
  id           INTEGER PRIMARY KEY,
  path         TEXT NOT NULL UNIQUE,
  lang         TEXT,
  grp          TEXT,
  profile      TEXT NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  size         INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  indexed_at   INTEGER NOT NULL
);

CREATE TABLE symbols (
  id             INTEGER PRIMARY KEY,
  file_id        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  name           TEXT NOT NULL,
  qualified_name TEXT NOT NULL,
  kind           TEXT NOT NULL,
  parent         TEXT,
  start_byte     INTEGER NOT NULL,
  end_byte       INTEGER NOT NULL,
  start_line     INTEGER NOT NULL,
  end_line       INTEGER NOT NULL,
  signature      TEXT NOT NULL,
  docstring      TEXT,
  content_hash   TEXT NOT NULL
);

CREATE TABLE imports (
  id      INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  module  TEXT NOT NULL,
  names   TEXT NOT NULL,
  alias   TEXT,
  line    INTEGER NOT NULL,
  kind    TEXT NOT NULL
);

CREATE TABLE import_targets (
  import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
  file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  PRIMARY KEY (import_id, file_id)
);

CREATE TABLE file_scopes (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  scope   TEXT NOT NULL,
  PRIMARY KEY (file_id, scope)
);

CREATE TABLE file_sides (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  side    TEXT NOT NULL CHECK (side IN ('client', 'server', 'shared')),
  PRIMARY KEY (file_id, side)
);

CREATE TABLE handler_meta (
  symbol_id INTEGER PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
  registrar TEXT NOT NULL
);

CREATE TABLE schema_nodes (
  id                INTEGER PRIMARY KEY,
  file_id           INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  key_path          TEXT NOT NULL,
  kind              TEXT NOT NULL,
  parent            TEXT,
  start_byte        INTEGER NOT NULL,
  end_byte          INTEGER NOT NULL,
  start_line        INTEGER NOT NULL,
  end_line          INTEGER NOT NULL,
  value_preview     TEXT,
  comment           TEXT,
  children_count    INTEGER NOT NULL DEFAULT 0,
  truncated_subtree INTEGER NOT NULL DEFAULT 0,
  uniformity        TEXT
);

CREATE TABLE doc_nodes (
  id         INTEGER PRIMARY KEY,
  file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  title      TEXT NOT NULL,
  slug       TEXT NOT NULL,
  level      INTEGER NOT NULL,
  kind       TEXT NOT NULL,
  parent     TEXT,
  start_byte INTEGER NOT NULL,
  end_byte   INTEGER NOT NULL,
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL,
  info       TEXT
);

CREATE TABLE chunks (
  id             INTEGER PRIMARY KEY,
  file_id        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind           TEXT NOT NULL,
  start_byte     INTEGER NOT NULL,
  end_byte       INTEGER NOT NULL,
  context_header TEXT NOT NULL DEFAULT '',
  body_mode      TEXT NOT NULL,
  oversize       INTEGER NOT NULL DEFAULT 0,
  content_hash   TEXT NOT NULL,
  part           INTEGER
);

CREATE TABLE chunk_symbols (
  chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  PRIMARY KEY (chunk_id, symbol_id)
);

CREATE TABLE symbol_refs (
  id         INTEGER PRIMARY KEY,
  file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  from_id    INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,
  name       TEXT NOT NULL,
  qualifier  TEXT,
  is_self    INTEGER NOT NULL,
  line       INTEGER NOT NULL,
  start_byte INTEGER NOT NULL
);

CREATE TABLE symbol_edges (
  from_id      INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  from_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  to_id        INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  to_file_id   INTEGER REFERENCES files(id) ON DELETE CASCADE,
  to_node_id   INTEGER REFERENCES schema_nodes(id) ON DELETE CASCADE,
  to_name      TEXT NOT NULL,
  rel          TEXT NOT NULL,
  resolution   TEXT NOT NULL CHECK (resolution IN ('exact', 'import', 'event', 'name')),
  line         INTEGER NOT NULL,
  detail       TEXT
);

CREATE TABLE annotations (
  chunk_id       INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  content_hash   TEXT NOT NULL,
  model_id       TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  purpose        TEXT NOT NULL,
  tags           TEXT NOT NULL,
  PRIMARY KEY (chunk_id, model_id, prompt_version)
);

CREATE TABLE vec_spaces (
  name     TEXT PRIMARY KEY,
  space    TEXT NOT NULL,
  model_id TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  status   TEXT NOT NULL CHECK (status IN ('building', 'live', 'retired'))
);

CREATE TABLE vectors_meta (
  chunk_id     INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  space_name   TEXT NOT NULL REFERENCES vec_spaces(name) ON DELETE CASCADE,
  content_hash TEXT NOT NULL,
  PRIMARY KEY (chunk_id, space_name)
);

CREATE VIRTUAL TABLE fts_code USING fts5(
  name, qualified_name, signature, docstring, body, annotation, path
);

CREATE TABLE packs (
  pack_id    TEXT PRIMARY KEY,
  query      TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  payload    TEXT NOT NULL
);

CREATE TABLE feedback (
  pack_id TEXT NOT NULL,
  ref     TEXT NOT NULL,
  signal  TEXT NOT NULL,
  ts      INTEGER NOT NULL
);

CREATE INDEX idx_symbols_name  ON symbols(name);
CREATE INDEX idx_symbols_qname ON symbols(qualified_name);
CREATE INDEX idx_symbols_file  ON symbols(file_id);
CREATE INDEX idx_imports_file  ON imports(file_id);
CREATE INDEX idx_schema_path   ON schema_nodes(key_path);
CREATE INDEX idx_schema_file   ON schema_nodes(file_id);
CREATE INDEX idx_doc_slug      ON doc_nodes(slug);
CREATE INDEX idx_doc_file      ON doc_nodes(file_id);
CREATE INDEX idx_chunks_file   ON chunks(file_id);
CREATE INDEX idx_refs_file     ON symbol_refs(file_id);
CREATE INDEX idx_edges_from    ON symbol_edges(from_id);
CREATE INDEX idx_edges_from_file ON symbol_edges(from_file_id);
CREATE INDEX idx_edges_to      ON symbol_edges(to_id);
CREATE INDEX idx_edges_to_file ON symbol_edges(to_file_id);
CREATE INDEX idx_edges_to_node ON symbol_edges(to_node_id);
CREATE INDEX idx_feedback_pack ON feedback(pack_id);

CREATE TABLE served_items (
  pack_id           TEXT NOT NULL,
  ref               TEXT NOT NULL,
  path              TEXT NOT NULL,
  answers_query     REAL NOT NULL,
  relevance         REAL NOT NULL,
  implementation    REAL NOT NULL,
  lexical_match     REAL NOT NULL,
  structural_value  REAL NOT NULL,
  graph_proximity   REAL NOT NULL,
  redundancy        REAL NOT NULL,
  ts                INTEGER NOT NULL
);

CREATE INDEX idx_served_pack ON served_items(pack_id);
CREATE INDEX idx_served_ts   ON served_items(ts);
