-- Stage 1 tables (tech spec 10.2). Created by stage1.static.Stage1DB.
CREATE TABLE IF NOT EXISTS personas (persona_id TEXT PRIMARY KEY, wave INTEGER, gender INTEGER,
  bf_source TEXT CHECK (bf_source IN ('measured','proxy','none')));
CREATE TABLE IF NOT EXISTS context_texts (persona_id TEXT PRIMARY KEY, text TEXT, lang TEXT,
  source TEXT CHECK (source IN ('user','event','market','none')));
CREATE TABLE IF NOT EXISTS context_demands (persona_id TEXT PRIMARY KEY, text_hash TEXT,
  c_star TEXT, match_how TEXT, D_style TEXT,             -- JSON, 12 values on -3..3
  D_O REAL, D_C REAL, D_E REAL, D_A REAL, D_ES REAL,     -- D~ in [-1, 1]
  Z TEXT, hits TEXT,                                     -- JSON: [{style, cls, span, toneless}]
  lexicon_ver TEXT, catalog_ver TEXT, artifact_hash TEXT, reader_status TEXT, rated_prior INTEGER);
CREATE TABLE IF NOT EXISTS demand_labels (text_id TEXT, text TEXT, lang TEXT, context_id TEXT,
  kind TEXT CHECK (kind IN ('bare','cue','minimal_pair','ooc')),
  rater_id TEXT, style TEXT, rating INTEGER CHECK (rating BETWEEN -3 AND 3),
  PRIMARY KEY (text_id, rater_id, style));
CREATE TABLE IF NOT EXISTS static_scores (pair_id TEXT, chooser TEXT, chosen TEXT, ctx_id TEXT,
  eta0 REAL, etaP_fit REAL, etaP_resid REAL, etaP_actor REAL, etaG REAL, eta REAL,
  a0 REAL, m0 REAL, fit REAL, fit_parts TEXT, mask TEXT, srrm_hash TEXT,
  PRIMARY KEY (pair_id, chooser));
CREATE TABLE IF NOT EXISTS unmatched_contexts (text_hash TEXT PRIMARY KEY, text TEXT, n_seen INTEGER);
CREATE TABLE IF NOT EXISTS artifacts (name TEXT PRIMARY KEY, sha256 TEXT, path TEXT, created_at TEXT);
