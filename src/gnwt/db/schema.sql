-- GNWT-TM database schema (Tech spec, section 3.2).
-- Every table also has config_version TEXT and created_at TEXT (ISO 8601, UTC).
-- Ground-truth labels are NOT stored in the database (sections 3.2 and 12.1).

CREATE TABLE IF NOT EXISTS personas (
  persona_id TEXT PRIMARY KEY, source TEXT, language TEXT,
  O REAL, C REAL, E REAL, A REAL, N REAL, json TEXT NOT NULL,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS pairs (            -- one date = one undirected pair
  pair_id TEXT PRIMARY KEY,                   -- "w<wave>_<smaller iid>_<larger iid>"
  wave INTEGER, i TEXT, j TEXT, context TEXT,
  split TEXT CHECK (split IN ('calib','test','reserve','pilot','handwritten')),
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS static_scores (    -- stage 1, (10)-(14)
  pair_id TEXT, chooser TEXT, chosen TEXT, S_P REAL, S_G REAL, a0 REAL,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (pair_id, chooser));

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,                -- sha1(pair_id|mode|seed|config_version)[:16]
  pair_id TEXT, mode TEXT CHECK (mode IN ('free','full')), seed INTEGER, opener TEXT,
  status TEXT CHECK (status IN ('queued','running','done','interrupted','failed')),
  rounds_done INTEGER, prompt_version TEXT, model_twin TEXT,
  started_at TEXT, finished_at TEXT, cost_usd REAL, error TEXT,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS turns (            -- every message spoken
  session_id TEXT, turn_idx INTEGER,          -- 0 is the opener, 1..2T are replies
  speaker TEXT, listener TEXT, t INTEGER,     -- t: speaker's reply number (1..T), 0 for the opener
  response TEXT, thought TEXT, self_affinity REAL,   -- self_affinity only in free mode
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, turn_idx));

CREATE TABLE IF NOT EXISTS cognitive_packets (  -- speaker's cognitive packet on reading the listener
  session_id TEXT, turn_idx INTEGER, twin TEXT,
  emotion_json TEXT, goal_json TEXT, planning_json TEXT, social_json TEXT,
  V REAL, Ar REAL, direction TEXT, reciprocity REAL, norm_risk TEXT,
  S_E REAL, S_GT REAL, S_P REAL, S_SN REAL, tau REAL, ignored TEXT,   -- ignored: "GT,P" ...
  D_AN REAL, D_EC REAL, D REAL, social_mask REAL, defensive_mask REAL, interrupted INTEGER,
  rag_doc_id TEXT, rag_l2sq REAL,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, turn_idx, twin));

CREATE TABLE IF NOT EXISTS affinity_logs (    -- (15)-(19), written in both modes
  session_id TEXT, twin TEXT, t INTEGER,
  psi_aff REAL, psi_dir REAL, psi_rec REAL, psi_coh REAL, psi_safe REAL,
  x REAL, kappa REAL, a_prev REAL, a REAL, e_target REAL, e_hat REAL,
  injected INTEGER,                           -- 1 if (a, e) was injected into the prompt (full mode)
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, twin, t));

CREATE TABLE IF NOT EXISTS session_outputs (  -- stage 2 result per direction, (21)-(23), (28)
  session_id TEXT, twin TEXT, partner TEXT,
  a0 REAL, aT REAL, S_I REAL, c_T REAL, delta_a REAL, t_star INTEGER, g_bar REAL,
  self_report_last REAL, interrupted INTEGER,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, twin));

CREATE TABLE IF NOT EXISTS readouts (         -- score of each arm for every directed decision
  pair_id TEXT, chooser TEXT, chosen TEXT, arm TEXT, score REAL,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (pair_id, chooser, arm));

CREATE TABLE IF NOT EXISTS warmth_scores (
  session_id TEXT, turn_idx INTEGER, rater TEXT,
  level INTEGER, value REAL, rationale TEXT,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, turn_idx, rater));

CREATE TABLE IF NOT EXISTS judge_scores (
  session_id TEXT, judge TEXT, chooser TEXT,
  J REAL, personality REAL, cognitive REAL, vibe REAL, goal REAL, hallucination INTEGER,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (session_id, judge, chooser));

CREATE TABLE IF NOT EXISTS api_calls (
  call_id TEXT PRIMARY KEY, session_id TEXT, purpose TEXT,
  model TEXT, prompt_hash TEXT, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, latency_ms INTEGER, attempt INTEGER, status TEXT, cached INTEGER,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS matching_runs (
  run_id TEXT PRIMARY KEY, population TEXT, solver TEXT,
  k INTEGER, p REAL, b INTEGER, objective REAL, blocking INTEGER, runtime_s REAL,
  config_version TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS matches (
  run_id TEXT, group_id INTEGER, member TEXT,
  config_version TEXT, created_at TEXT,
  PRIMARY KEY (run_id, group_id, member));

CREATE INDEX IF NOT EXISTS idx_api_calls_session ON api_calls (session_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_prompt ON api_calls (prompt_hash);
