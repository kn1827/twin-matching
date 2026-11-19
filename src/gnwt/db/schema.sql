-- GNWT-TM database, Stage 1 rewrite (stage1.md, section 10.2).
-- Every table also has config_version TEXT and created_at TEXT (ISO 8601, UTC).
-- Ground-truth decisions are NOT stored here: only predictions and their provenance.

CREATE TABLE IF NOT EXISTS personas (
  persona_id TEXT PRIMARY KEY, source TEXT, market TEXT, language TEXT,
  O REAL, C REAL, E REAL, A REAL, N REAL,
  bf_source TEXT CHECK (bf_source IN ('measured','proxy','none')),
  json TEXT NOT NULL,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS pairs (               -- one undirected pair (a date)
  pair_id TEXT PRIMARY KEY, market TEXT, wave INTEGER, i TEXT, j TEXT, split TEXT,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS context_texts (
  persona_id TEXT PRIMARY KEY, text TEXT, lang TEXT,
  source TEXT CHECK (source IN ('user','event','market','none')),
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS context_demands (
  persona_id TEXT PRIMARY KEY, text_hash TEXT,
  c_star TEXT, match_how TEXT, D_style TEXT,             -- JSON, 12 values on -3..3
  D_O REAL, D_C REAL, D_E REAL, D_A REAL, D_ES REAL,     -- D~ in [-1, 1]
  Z TEXT, hits TEXT,                                     -- JSON: [{style, cls, span, toneless}]
  lexicon_ver TEXT, catalog_ver TEXT, artifact_hash TEXT,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS demand_labels (
  text_id TEXT, text TEXT, lang TEXT, context_id TEXT,
  kind TEXT CHECK (kind IN ('bare','cue','minimal_pair','ooc')),
  rater_id TEXT, style TEXT, rating INTEGER CHECK (rating BETWEEN -3 AND 3),
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (text_id, rater_id, style));

CREATE TABLE IF NOT EXISTS static_scores (
  pair_id TEXT, chooser TEXT, chosen TEXT, ctx_id TEXT,
  eta0 REAL, etaP_fit REAL, etaP_resid REAL, etaP_actor REAL, etaG REAL, eta REAL,
  a0 REAL, m0 REAL, fit REAL, fit_parts TEXT, mask TEXT, srrm_hash TEXT,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (pair_id, chooser));

CREATE TABLE IF NOT EXISTS unmatched_contexts (  -- for offline catalog review
  text_hash TEXT PRIMARY KEY, text TEXT, n_seen INTEGER,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS artifacts (
  name TEXT PRIMARY KEY, sha256 TEXT, path TEXT,
  config_version TEXT, created_at TEXT);

CREATE INDEX IF NOT EXISTS idx_static_chooser ON static_scores (chooser);
