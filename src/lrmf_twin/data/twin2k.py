"""T0 data book for Twin-2K-500: person x item matrix of binary targets, raw answers, demographics, item text.

Binarisation rules [quy ước] (the design leaves the target definition open: "nhị phân hoặc từng lựa chọn"):

* Demographics block (QID11-QID24) -> person features (one-hot), never targets.
* Dropped: descriptive blocks (DB), every text entry (TE: free text, numeric estimates, Forward Flow) and the
  multi-select Wason card task (MAHR QID221).
* Ordinal items: Matrix rows (Likert, Bipolar), sliders, BDI multi-select items (value = most severe statement
  ticked) and every single-choice MC outside the cognitive tests. Target y = 1{value > c}, where c is the cut
  point between scale points (sliders: 10, 20, ..., 90) whose positive rate is closest to 0.5. Two-option
  questions use y = 1{first option} so that "Yes / True / Yes, I would purchase" is the positive class.
* Nominal items: single-choice MC in the cognitive tests with >= 3 options; y = 1{modal option} (in practice
  the correct answer).
* The cut point and the modal option use only the item's marginal distribution, never a person's own answer,
  and are the same for every model compared. Items with < 200 answers or a positive rate outside
  [0.05, 0.95] are dropped.

Only wave 1-3 answers are used; wave 4 repeats wave 1-3 questions (test-retest) and would leak.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lrmf_twin.data.fetch import DEFAULT_DEST, ensure

DEMOGRAPHIC_NAMES = {
    "QID11": "region", "QID12": "sex", "QID13": "age", "QID14": "education", "QID15": "race",
    "QID16": "US citizen", "QID17": "marital status", "QID18": "religion", "QID19": "religious attendance",
    "QID20": "party", "QID21": "family income", "QID22": "political views", "QID23": "household size",
    "QID24": "employment",
}
BDI_STEM = ("Beck Depression Inventory: pick the statement that best describes the way you have been "
            "feeling during the past two weeks.")
MIN_ANSWERS = 200
RATE_RANGE = (0.05, 0.95)
STEM_MAX_CHARS = 700


def clean(text: str | None) -> str:
    text = (text or "").replace("�", "'").replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def _jsonable(o):
    """json.dumps fallback for numpy scalars and arrays."""
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


@dataclass
class Item:
    item_id: str                 # CSV column (MAVR items: the QID)
    qid: str
    block: str
    kind: str                    # "ordinal" | "nominal"
    stem: str
    row: str | None
    options: list[str]           # labels of answer positions 1..K (sliders: two bins around the cut)
    positive: list[int]          # 1-based positions that make y = 1
    cut: float | None            # ordinal cut point on the raw value scale
    is_slider: bool = False

    @property
    def group(self) -> str:
        """Question block: items of one Qualtrics question share it (matrix rows, slider statements)."""
        return f"{self.block}|{self.qid}"

    def question(self) -> str:
        return self.stem if not self.row else f"{self.stem} | {self.row}"

    def positive_labels(self) -> list[str]:
        return [self.options[k - 1] for k in self.positive]

    def answer_label(self, value: float) -> str:
        if np.isnan(value):
            return "(no answer)"
        if self.is_slider:
            return f"{value:.0f} (scale 0-100)"
        k = int(round(value))
        return self.options[k - 1] if 1 <= k <= len(self.options) else str(value)

    def text(self) -> str:
        """Text used by the T2 embedder: block, question, options and the positive set."""
        opts = "; ".join(self.options)
        return (f"{self.block}. {self.question()} Options: {opts}. "
                f"Target answer: {' or '.join(self.positive_labels())}.")[:1500]


@dataclass
class SurveyData:
    pids: np.ndarray             # (P,)
    items: list[Item]            # (J,)
    Y: np.ndarray                # (P, J) float32 binary target, NaN = not asked / no answer
    V: np.ndarray                # (P, J) float32 raw value (position or slider value), NaN = missing
    X_demo: np.ndarray           # (P, D) float32 one-hot demographics
    demo_names: list[str]
    demo_text: list[str]         # (P,) one line per person for prompts
    source: dict = field(default_factory=dict)
    wave: np.ndarray | None = None   # (P, J) int16 wave / period of each answered cell, -1 = none (R3 needs it)
    name: str = "twin2k"

    @property
    def shape(self) -> tuple[int, int]:
        return self.Y.shape

    def subset(self, persons: np.ndarray | None = None, items: np.ndarray | None = None) -> "SurveyData":
        pi = np.arange(len(self.pids)) if persons is None else np.asarray(persons)
        ji = np.arange(len(self.items)) if items is None else np.asarray(items)
        wave = None if self.wave is None else self.wave[np.ix_(pi, ji)]
        return SurveyData(self.pids[pi], [self.items[j] for j in ji], self.Y[np.ix_(pi, ji)],
                          self.V[np.ix_(pi, ji)], self.X_demo[pi], self.demo_names,
                          [self.demo_text[p] for p in pi], dict(self.source), wave, self.name)

    # -- cache of a processed dataset (the raw GSS file takes minutes to parse) --------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"items": [asdict(it) for it in self.items], "demo_names": self.demo_names,
                "demo_text": self.demo_text, "source": self.source, "name": self.name}
        arrays = {"pids": self.pids, "Y": self.Y, "V": self.V, "X_demo": self.X_demo,
                  "meta": np.frombuffer(json.dumps(meta, default=_jsonable).encode("utf-8"), dtype=np.uint8)}
        if self.wave is not None:
            arrays["wave"] = self.wave
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load_cached(path: Path) -> "SurveyData":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        items = [Item(**it) for it in meta["items"]]
        return SurveyData(z["pids"], items, z["Y"], z["V"], z["X_demo"], meta["demo_names"], meta["demo_text"],
                          meta["source"], z["wave"] if "wave" in z.files else None, meta["name"])


def _labels_by_column(q: dict, labels_key: str, ids_key: str) -> dict[str, str]:
    """Map each CSV column `{QID}_{id}` to its row/statement label (ids missing -> catalog order)."""
    labels = q.get(labels_key, []) or []
    ids = q.get(ids_key) or []
    cols = q.get("csv_columns", [])
    if ids and len(ids) == len(labels):
        by_id = dict(zip(ids, labels))
        return {c: by_id.get(c.rsplit("_", 1)[-1], "") for c in cols}
    return dict(zip(cols, labels))


def _balanced_cut(values: np.ndarray, cuts: np.ndarray) -> float:
    rates = np.array([(values > c).mean() for c in cuts])
    return float(cuts[int(np.argmin(np.abs(rates - 0.5)))])


def _ordinal_item(item_id, qid, block, stem, row, options, values, two_option_first=True) -> tuple[Item, np.ndarray]:
    k = len(options)
    obs = values[~np.isnan(values)]
    if k == 2 and two_option_first:
        cut, positive = 1.5, [1]
        y = np.where(np.isnan(values), np.nan, (values < cut).astype(np.float32))
    else:
        cut = _balanced_cut(obs, np.arange(1.5, k, 1.0)) if obs.size else 1.5
        positive = [p for p in range(1, k + 1) if p > cut]
        y = np.where(np.isnan(values), np.nan, (values > cut).astype(np.float32))
    return Item(item_id, qid, block, "ordinal", stem, row, options, positive, cut), y


def load(dest: Path = DEFAULT_DEST) -> SurveyData:
    dest = ensure(dest)
    catalog = json.loads((dest / "question_catalog.json").read_text(encoding="utf-8"))
    df = pd.read_csv(dest / "wave1_3_response.csv", low_memory=False).sort_values("pid").reset_index(drop=True)
    pids = df["pid"].to_numpy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)  # noqa: E731

    items: list[Item] = []
    ys: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    demo_cols, demo_names, demo_labels = [], [], {}

    for q in catalog:
        qtype = q.get("QuestionType")
        sel = (q.get("Settings") or {}).get("Selector")
        block = clean(q.get("BlockName"))
        qid = q["QuestionID"]
        cols = [c for c in q.get("csv_columns", []) if c in df.columns]
        if not cols or qtype in ("DB", "TE") or sel == "MAHR":
            continue
        stem = clean(q.get("QuestionText"))[:STEM_MAX_CHARS]

        if block == "Demographics":
            opts = [clean(o) for o in q.get("Options", [])]
            v = num(cols[0])
            name = DEMOGRAPHIC_NAMES.get(qid, stem[:40])
            demo_labels[name] = (v, opts)
            for k, o in enumerate(opts, start=1):
                demo_cols.append((v == k).astype(np.float32))
                demo_names.append(f"{name}={o}")
            continue

        if qtype == "MC" and sel == "MAVR":           # BDI: value = most severe statement ticked
            opts = [clean(o) for o in q.get("Options", [])]
            mat = np.stack([num(c) for c in cols], axis=1)
            ticked = ~np.isnan(mat)
            v = np.where(ticked.any(1), np.where(ticked, np.arange(1, mat.shape[1] + 1), 0).max(1), np.nan)
            item, y = _ordinal_item(qid, qid, block, stem or BDI_STEM, None, opts, v)
            items.append(item); ys.append(y); vs.append(v)
            continue

        if qtype == "MC":
            opts = [clean(o) for o in q.get("Options", [])]
            v = num(cols[0])
            if "Cognitive" in block and len(opts) >= 3:
                obs = v[~np.isnan(v)].astype(int)
                mode = int(np.bincount(obs).argmax()) if obs.size else 1
                y = np.where(np.isnan(v), np.nan, (v == mode).astype(np.float32))
                items.append(Item(cols[0], qid, block, "nominal", stem, None, opts, [mode], None))
            else:
                item, y = _ordinal_item(cols[0], qid, block, stem, None, opts, v)
                items.append(item)
            ys.append(y); vs.append(v)
            continue

        if qtype == "Matrix":
            opts = [clean(o) for o in q.get("Columns", [])]
            row_by_col = _labels_by_column(q, "Rows", "RowsID")
            for c in cols:
                row = clean(row_by_col.get(c, ""))
                row_opts = opts
                # Bipolar price lists: columns are "1"/"2", the row holds "left option:right option"
                if opts == ["1", "2"] and row.count(":") == 1:
                    row_opts = [s.strip() for s in row.split(":")]
                    row = f"left option: {row_opts[0]}; right option: {row_opts[1]}"
                v = num(c)
                item, y = _ordinal_item(c, qid, block, stem, row or None, row_opts, v)
                items.append(item); ys.append(y); vs.append(v)
            continue

        if qtype == "Slider":
            st_by_col = _labels_by_column(q, "Statements", "StatementsID")
            for c in cols:
                row = clean(st_by_col.get(c, "")) or None
                v = num(c)
                obs = v[~np.isnan(v)]
                cut = _balanced_cut(obs, np.arange(10.0, 100.0, 10.0)) if obs.size else 50.0
                y = np.where(np.isnan(v), np.nan, (v > cut).astype(np.float32))
                opts = [f"at most {cut:.0f} (on a 0-100 scale)", f"more than {cut:.0f} (on a 0-100 scale)"]
                items.append(Item(c, qid, block, "ordinal", stem, row, opts, [2], cut, is_slider=True))
                ys.append(y); vs.append(v)

    Y = np.stack(ys, axis=1).astype(np.float32)
    V = np.stack(vs, axis=1).astype(np.float32)
    n_obs = (~np.isnan(Y)).sum(0)
    rate = np.nanmean(Y, axis=0)
    keep = (n_obs >= MIN_ANSWERS) & (rate >= RATE_RANGE[0]) & (rate <= RATE_RANGE[1])
    keep_idx = np.flatnonzero(keep)
    items = [items[j] for j in keep_idx]
    Y, V = Y[:, keep_idx], V[:, keep_idx]

    X_demo = np.stack(demo_cols, axis=1).astype(np.float32)
    demo_text = []
    for p in range(len(pids)):
        parts = []
        for name, (v, opts) in demo_labels.items():
            k = v[p]
            if not np.isnan(k) and 1 <= int(k) <= len(opts):
                parts.append(f"{name}: {opts[int(k) - 1]}")
        demo_text.append("; ".join(parts))

    manifest = {}
    if (dest / "manifest.json").exists():
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    source = {"dataset": "Twin-2K-500", "dir": str(dest), "sha256": manifest.get("sha256", {}),
              "items_kept": int(keep.sum()), "items_dropped": int((~keep).sum())}
    return SurveyData(pids, items, Y, V, X_demo, demo_names, demo_text, source)
