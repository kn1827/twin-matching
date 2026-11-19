"""U5-U8, U12 and the worked example of stage1.md section 3.5."""

from __future__ import annotations

import numpy as np
import pytest

from gnwt.stage1.context.demand import ReaderArtifacts, extract_demands
from gnwt.stage1.context.styles import CUE_CLASSES, STYLES

# (text, expected [(style, class), ...]) - U6, >= 40 cases, vi and en
U6_CASES = [
    ("kỷ luật", [("DEP", "base")]),
    ("rất cần kỷ luật", [("DEP", "strong")]),
    ("không cần kỷ luật", [("DEP", "neg")]),
    ("kỷ luật không quan trọng", [("DEP", "neg")]),
    ("không được đi muộn", [("DEP", "strong")]),
    ("không cạnh tranh", [("COO", "base")]),
    ("rất cần kỷ luật và đúng giờ", [("DEP", "strong"), ("DEP", "strong")]),
    ("cần sáng tạo, nhưng không cần tỉ mỉ", [("INN", "base"), ("DET", "neg")]),
    ("sáng tạo thì không cần", [("INN", "neg")]),
    ("hơi cần cẩn thận", [("DET", "weak")]),
    ("không thể thiếu kỷ luật", [("DEP", "strong")]),
    ("kỷ luật là không thể thiếu", [("DEP", "strong")]),
    ("không chỉ sáng tạo mà còn đúng giờ", [("INN", "base"), ("DEP", "base")]),
    ("kỷ luật là bắt buộc", [("DEP", "strong")]),
    ("đúng giờ rất quan trọng", [("DEP", "strong")]),
    ("bắt buộc đúng giờ", [("DEP", "strong")]),
    ("tuỳ hứng", [("DEP", "rev")]),
    ("tùy hứng", [("DEP", "rev")]),
    ("cạnh tranh", [("COO", "rev")]),
    ("không nóng nảy", [("SCT", "base")]),
    ("đừng đến muộn", [("DEP", "strong")]),
    ("không cần quá tỉ mỉ", [("DET", "weak")]),
    ("cần tinh thần đồng đội", [("COO", "base")]),
    ("hợp tác, lắng nghe và bình tĩnh", [("COO", "base"), ("EMP", "base"), ("SCT", "base")]),
    ("chịu áp lực cao", [("STR", "base")]),
    ("không khí vui vẻ", [("SOC", "base")]),
    ("nhóm ôn thi xác suất", []),
    ("very disciplined", [("DEP", "strong")]),
    ("must be punctual", [("DEP", "strong")]),
    ("no need to be creative", [("INN", "neg")]),
    ("creativity is not important", [("INN", "neg")]),
    ("don't be late", [("DEP", "strong")]),
    ("not competitive", [("COO", "base")]),
    ("competitive", [("COO", "rev")]),
    ("not only creative but also punctual", [("INN", "base"), ("DEP", "base")]),
    ("punctuality is a must", [("DEP", "strong")]),
    ("somewhat careful", [("DET", "weak")]),
    ("creative, but no need to be careful", [("INN", "base"), ("DET", "neg")]),
    ("we need teamwork and patience", [("COO", "base"), ("SCT", "base")]),
    ("very social and outgoing", [("SOC", "strong"), ("SOC", "strong")]),
    ("calm under pressure", [("SCT", "base"), ("STR", "base")]),
    ("never impulsive", [("SCT", "strong")]),
    ("leadership is optional", [("LEA", "neg")]),
]


def test_u6_rule_table(lexicon):
    assert len(U6_CASES) >= 40
    bad = []
    for text, expected in U6_CASES:
        _, hits = lexicon.read(text)
        got = [(h.style, h.cls) for h in hits]
        if got != expected:
            bad.append((text, got, expected))
    assert not bad, bad


def test_spec_example_section_3_5(reader_illustrative):
    cd = extract_demands("nhóm ôn thi xác suất, rất cần kỷ luật, không cần sáng tạo", None, reader_illustrative)
    assert cd.match.c_star == "study_group"
    assert [(h.style, h.cls) for h in cd.hits] == [("DEP", "strong"), ("INN", "neg")]
    assert cd.D_style[STYLES.index("DEP")] == pytest.approx(2.30, abs=0.005)
    assert cd.D_style[STYLES.index("INN")] == pytest.approx(-0.1667, abs=0.001)
    np.testing.assert_allclose(cd.D_tilde, [0.20, 0.59, 0.17, 0.50, 0.33], atol=0.006)
    bare = extract_demands("học nhóm", None, reader_illustrative)
    np.testing.assert_allclose(bare.D_tilde, [0.28, 0.50, 0.17, 0.50, 0.33], atol=0.006)


def test_u5_defaults(reader_illustrative, catalog_illustrative):
    cd = extract_demands("ôn thi cuối kỳ", None, reader_illustrative)
    assert not cd.hits
    assert np.array_equal(cd.D_style, catalog_illustrative.prior(cd.match))
    cd2 = extract_demands("đi chơi cuối tuần", None, reader_illustrative)
    assert cd2.match.how == "generic" and not cd2.hits
    assert np.array_equal(cd2.D_style, catalog_illustrative.generic)


def test_u7_toneless(reader_illustrative):
    a = extract_demands("học nhóm, cần kỷ luật", None, reader_illustrative)
    b = extract_demands("hoc nhom, can ky luat", None, reader_illustrative)
    assert a.match.c_star == b.match.c_star == "study_group"
    assert [(h.style, h.cls) for h in a.hits] == [(h.style, h.cls) for h in b.hits]
    assert b.match.how == "toneless" and all(h.toneless for h in b.hits)
    # a toned word never matches a toneless form ("hỏi" is not "hơi")
    _, hits = reader_illustrative.lexicon.read("hỏi cẩn thận")
    assert [(h.style, h.cls) for h in hits] == [("DET", "base")]


def test_u8_monotone(reader_illustrative):
    th = reader_illustrative.theta
    assert th["neg"] <= th["weak"] <= th["base"] <= th["strong"]
    for cue, pre in (("kỷ luật", "DEP"), ("sáng tạo", "INN"), ("hợp tác", "COO"), ("bình tĩnh", "SCT")):
        vals = []
        for tmpl in ("rất cần {}", "cần {}", "hơi cần {}", "không cần {}"):
            for ctx in ("học nhóm", "hẹn hò", ""):
                cd = extract_demands(f"{ctx}, {tmpl.format(cue)}" if ctx else tmpl.format(cue), None, reader_illustrative)
                vals.append((tmpl, ctx, cd.D_style[STYLES.index(pre)]))
        for ctx in ("học nhóm", "hẹn hò", ""):
            seq = [v for t, c, v in vals if c == ctx]
            assert seq == sorted(seq, reverse=True), (cue, ctx, seq)


def test_u12_deterministic(reader_illustrative):
    a = extract_demands("rất cần kỷ luật, không cần sáng tạo", None, reader_illustrative)
    b = extract_demands("rất cần kỷ luật, không cần sáng tạo", None, reader_illustrative)
    assert a.D_tilde.tobytes() == b.D_tilde.tobytes()
    assert a.artifact_hash == b.artifact_hash and a.text_hash == b.text_hash


def test_unestimated_reader_keeps_default(lexicon, catalog_illustrative):
    art = ReaderArtifacts.unestimated(lexicon, catalog_illustrative)
    cd = extract_demands("học nhóm, rất cần kỷ luật", None, art)
    assert "reader_unestimated" in cd.flags
    assert np.array_equal(cd.D_style, catalog_illustrative.prior(cd.match))


def test_no_text_uses_event(reader_illustrative):
    cd = extract_demands(None, "speed dating event", reader_illustrative)
    assert cd.source == "event" and cd.match.c_star == "speed_dating" and "no_text" in cd.flags
    assert np.all(cd.D_tilde == 0)          # unrated context: neutral profile


def test_lexicon_loader_checks(lexicon):
    assert lexicon.collisions == {}
    assert lexicon.n_entries >= 12 * 2 * 5
    assert set(CUE_CLASSES) == {"neg", "rev", "weak", "base", "strong"}
