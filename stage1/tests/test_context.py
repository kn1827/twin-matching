"""Context reader: rule table (U6), accent-folded matching (U7), monotonicity (U8), prior behaviour (U5),
determinism (U12), the worked example of tech spec 3.5, and the lexicon loader checks."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from stage1.context.demand import extract_demands
from stage1.context.lexicon import Lexicon, resolve_class
from stage1.context.styles import DOMAINS, STYLES, domain_matrix
from stage1.context.text import split_clauses

# (text, expected [(style, class), ...] in reading order)  -- U6, >= 40 cases
RULE_CASES = [
    ("discipline", [("DEP", "base")]),
    ("need discipline", [("DEP", "base")]),
    ("really need discipline", [("DEP", "strong")]),
    ("must be punctual", [("DEP", "strong")]),
    ("no need for discipline", [("DEP", "neg")]),
    ("discipline is not important", [("DEP", "neg")]),
    ("discipline doesn't matter", [("DEP", "neg")]),
    ("don't be late", [("DEP", "strong")]),
    ("never late", [("DEP", "strong")]),
    ("no competition", [("COO", "base")]),
    ("not competitive", [("COO", "base")]),
    ("competitive", [("COO", "rev")]),
    ("very competitive", [("COO", "rev")]),
    ("very disciplined and punctual", [("DEP", "strong"), ("DEP", "strong")]),
    ("need creativity, but no need to be meticulous", [("INN", "base"), ("DET", "neg")]),
    ("not only punctual but also creative", [("DEP", "base"), ("INN", "base")]),
    ("punctuality is a must", [("DEP", "strong")]),
    ("teamwork is essential", [("COO", "strong")]),
    ("we can't do without teamwork", [("COO", "strong")]),
    ("somewhat careful", [("DET", "weak")]),
    ("a bit more creative", [("INN", "weak")]),
    ("not too intense", [("STR", "weak")]),
    ("creativity is optional", [("INN", "neg")]),
    ("no need to be creative", [("INN", "neg")]),
    ("doesn't have to be a leader", [("LEA", "neg")]),
    ("extremely curious people", [("CUR", "strong")]),
    ("dig deeper into the topic", [("CUR", "base")]),
    ("open ended problems", [("AMB", "base")]),
    ("structured plan please", [("AMB", "rev")]),
    ("we want to make friends", [("SOC", "base")]),
    ("quiet study", [("SOC", "rev")]),
    ("someone to take the lead", [("LEA", "base")]),
    ("help each other", [("COO", "base")]),
    ("kind and patient", [("EMP", "base"), ("SCT", "base")]),
    ("stay calm under pressure", [("SCT", "base"), ("STR", "base")]),
    ("high pressure deadlines", [("STR", "base"), ("DEP", "base")]),
    ("stress free and relaxed", [("STR", "rev"), ("STR", "rev")]),
    ("avoid hot headed people", [("SCT", "strong")]),
    ("not impulsive", [("SCT", "base")]),
    ("just for fun", [("ACH", "rev")]),
    ("aim for top grades", [("ACH", "base")]),
    ("really need discipline. creativity is not important", [("DEP", "strong"), ("INN", "neg")]),
    ("brainstorming with fresh ideas", [("INN", "base"), ("INN", "base")]),
    ("however, very reliable", [("DEP", "strong")]),
]


@pytest.mark.parametrize("text,expected", RULE_CASES)
def test_u6_rule_table(lexicon, text, expected):
    cl = split_clauses(text, lexicon.punct, lexicon.split_words, lexicon.keep_bigrams)
    got = [(h.style, h.cls) for h in lexicon.read(cl)]
    assert got == expected


def test_u6_has_at_least_40_cases():
    assert len(RULE_CASES) >= 40


def test_resolve_class_table():
    assert resolve_class(+1, None, False, False) == "base"
    assert resolve_class(+1, "strong", False, False) == "strong"
    assert resolve_class(+1, None, True, False) == "neg"
    assert resolve_class(-1, "prohibit", False, False) == "strong"
    assert resolve_class(-1, "negator", False, False) == "base"
    assert resolve_class(-1, None, False, False) == "rev"


def test_u7_accent_folded_match(art):
    plain = extract_demands("study group, really need discipline", None, art)
    accented = extract_demands("stúdy groùp, réally need díscipline", None, art)
    assert accented.match.c_star == plain.match.c_star == "study_group"
    assert accented.match.how == "toneless" and plain.match.how == "exact"
    assert [(h.style, h.cls) for h in accented.hits] == [(h.style, h.cls) for h in plain.hits]
    assert all(h.toneless for h in accented.hits) and not any(h.toneless for h in plain.hits)


def test_u5_prior_only(art):
    cat = art.catalog
    bare = extract_demands("study group", None, art)
    assert np.array_equal(bare.D_style, cat.profile_of("study_group"))
    nothing = extract_demands("let us meet on tuesday", None, art)
    assert nothing.match.how == "generic"
    assert np.array_equal(nothing.D_style, cat.generic)
    cue_only = extract_demands("need creativity", None, art)          # cue without a context: generic prior
    s = STYLES.index("INN")
    assert cue_only.match.how == "generic"
    Z = 1 / (1 + art.k[s])
    assert cue_only.D_style[s] == pytest.approx(Z * art.theta[s, 3] + (1 - Z) * cat.generic[s])


def test_spec_worked_example(art):
    """Tech spec 3.5 with its illustrative numbers: D_hat_DEP = 2.30, D_hat_INN = -0.17,
    D~ = (O 0.20, C 0.59, E 0.17, A 0.50, ES 0.33); bare 'study group' -> (0.28, 0.50, 0.17, 0.50, 0.33)."""
    cd = extract_demands("exam prep group for probability, really need discipline, no need for creativity", None, art)
    assert cd.match.c_star == "study_group"
    assert cd.D_style[STYLES.index("DEP")] == pytest.approx(2.30, abs=0.005)
    assert cd.D_style[STYLES.index("INN")] == pytest.approx(-0.1667, abs=0.001)
    assert np.allclose(cd.D_tilde, [0.20, 0.59, 0.17, 0.50, 0.33], atol=0.01)
    bare = extract_demands("study group", None, art)
    assert np.allclose(bare.D_tilde, [0.28, 0.50, 0.17, 0.50, 0.33], atol=0.01)
    assert list(DOMAINS) == ["O", "C", "E", "A", "ES"]


KEYWORD_OF = {"DEP": "punctual", "DET": "careful", "ACH": "ambitious", "INN": "creative", "CUR": "curious",
              "AMB": "flexible", "SOC": "sociable", "LEA": "leadership", "COO": "teamwork", "EMP": "empathy",
              "STR": "pressure", "SCT": "calm"}


def test_u8_monotone_levels(art):
    for cid, entry in art.catalog.contexts.items():
        alias = entry.aliases[0]
        for s, kw in KEYWORD_OF.items():
            i = STYLES.index(s)
            vals = []
            for tmpl in ("{a}, very {k}", "{a}, {k}", "{a}, somewhat {k}", "{a}, no need for {k}"):
                cd = extract_demands(tmpl.format(a=alias, k=kw), None, art)
                last = [h.style for h in cd.hits if h.clause == 1]     # clause 0 is the alias
                assert last == [s], (tmpl, kw, cd.hits)
                vals.append(cd.D_style[i])
            assert vals[0] >= vals[1] >= vals[2] >= vals[3], (cid, s, vals)


def test_u12_deterministic(art):
    t = "project team, really need discipline and fresh ideas, no competition"
    a, b = extract_demands(t, None, art), extract_demands(t, None, art)
    assert a.D_tilde.tobytes() == b.D_tilde.tobytes() and a.artifact_hash == b.artifact_hash


def test_event_and_market_fallback(art):
    ev = extract_demands(None, "speed dating event", art)
    assert ev.source == "event" and ev.match.c_star == "speed_dating" and ev.match.how == "event"
    mk = extract_demands("", None, art, market="running")
    assert mk.source == "market" and mk.match.c_star == "running"
    none = extract_demands(None, None, art)
    assert none.source == "none" and none.match.how == "generic"


def test_domain_matrix_rows():
    A = domain_matrix()
    assert np.allclose(A.sum(1), 1 / 3)
    keep = np.ones(12, bool)
    keep[STYLES.index("STR")] = keep[STYLES.index("SCT")] = False
    assert np.all(domain_matrix(keep)[DOMAINS.index("ES")] == 0)          # whole domain dropped -> neutral


def test_unrated_catalog_is_neutral(lexicon):
    from stage1.context.catalog import Catalog
    from stage1.context.demand import ReaderArtifacts
    from tests.conftest import CAT
    art0 = ReaderArtifacts.illustrative(lexicon, Catalog.load(CAT))
    cd = extract_demands(None, "speed dating event", art0)
    assert not cd.rated_prior and np.all(cd.D_tilde == 0)


def _lex(tmp_path, entries, modifiers=None):
    raw = {"version": "t", "entries": entries, "modifiers": modifiers or {}}
    p = tmp_path / "lex.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_loader_rejects_bad_lexicons(tmp_path):
    full = [{"style": s, "pole": 1, "en": [f"{s.lower()}{k}" for k in range(5)]} for s in STYLES]
    Lexicon.load(_lex(tmp_path, full))
    short = [dict(e) for e in full]
    short[0] = {"style": "DEP", "pole": 1, "en": ["a", "b"]}
    with pytest.raises(ValueError, match="fewer than"):
        Lexicon.load(_lex(tmp_path, short))
    dup = full + [{"style": "DET", "pole": 1, "en": ["dep0"]}]
    with pytest.raises(ValueError, match="two styles"):
        Lexicon.load(_lex(tmp_path, dup))
    p = tmp_path / "bool.yaml"
    p.write_text("version: t\nentries:\n" + "".join(
        f"  - {{style: {s}, pole: 1, en: [{s.lower()}0, {s.lower()}1, {s.lower()}2, {s.lower()}3, {s.lower()}4]}}\n"
        for s in STYLES) + "modifiers:\n  negator: {pos: pre, en: [not, no]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a string"):
        Lexicon.load(p)


def test_locked_hash_mismatch(tmp_path):
    from tests.conftest import LEX
    with pytest.raises(ValueError, match="sha256"):
        Lexicon.load(LEX, sha256="0" * 64)
