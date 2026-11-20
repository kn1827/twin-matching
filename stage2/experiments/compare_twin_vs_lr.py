"""So sánh twin "sửa phần dư của nền LR + MF" với baseline logistic regression.

    python experiments/compare_twin_vs_lr.py                       # Twin-2K, R1-R6, Ollama qwen2.5:3b
    python experiments/compare_twin_vs_lr.py --dataset gss         # GSS 2000-2024 (tải ~48 MB lần đầu)
    python experiments/compare_twin_vs_lr.py --dataset movielens   # MovieLens-1M
    python experiments/compare_twin_vs_lr.py --quick               # chạy thử nhanh (ít người, ít ô LLM)
    python experiments/compare_twin_vs_lr.py --llm none            # chỉ phần thống kê (LR, T1, T2)
    python experiments/compare_twin_vs_lr.py --regimes R1,R3 --llm-cells 100 --t5 a

Chế độ: R1 dày, R2 câu mới, R3 đợt khuyết (khối câu x năm/quý; Twin-2K không có -> để trống), R4 người
thưa, R5 người mới (chỉ nhân khẩu), R6 câu lạnh (k câu trả lời). Kết quả: result/<thời điểm>_<bộ dữ liệu>[_tag]/
gồm overview.json (số liệu gọn cho bảng tổng hợp nhiều bộ dữ liệu), summary.md (các bảng in ra màn hình), metrics.csv, tests.json,
level2_cells.csv (ô có gọi LLM, kèm D), prompts.jsonl, config.json; --save-predictions ghi thêm dự đoán
từng ô của mọi mô hình.

Hai tầng so sánh, mỗi chế độ một bảng:
* Tầng 1 - mọi ô trong phạm vi chế độ: LR-demo, LR (baseline chính: nhân khẩu + câu trả lời khác của người
  đó), B0, B2 = T1 (LR + MF), B3 = +T2, B4 = +T3 (chỉ R4, cần LLM).
* Tầng 2 - mẫu ô có gọi LLM (--llm-cells mỗi chế độ): thêm T-b (twin đứng riêng), B5 (D thô, stacking không
  cổng), B6 (+D~), B7 (+cổng), B8 (T5b), B9 (super learner + T9).
ΔAUC / Δlog-loss so với LR có CI 95% bootstrap ghép cặp theo người. Kiểm định chính: LR test của β trong
logit p = η_B4 + β·D~ (Holm trên chế độ x biến thể). Ghi công: thêm sau, thêm trước, Shapley 2 nhóm.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrmf_twin import __version__  # noqa: E402
from lrmf_twin.data import DATASETS, load_dataset  # noqa: E402
from lrmf_twin.llm import DEFAULT_LLM, LLMClient, LLMUnavailable  # noqa: E402
from lrmf_twin.metrics import corr_ci, logloss_cells, lr_test, paired_bootstrap, summary  # noqa: E402
from lrmf_twin.pipeline import LEVEL1_MODELS, RunConfig, run_level2, run_regime  # noqa: E402
from lrmf_twin.regimes import REGIMES, supported  # noqa: E402
from lrmf_twin.t1_base import BaseConfig  # noqa: E402
from lrmf_twin.t2_encoder import DEFAULT_EMBEDDER, embed_items  # noqa: E402

REF = "LR"
LEVEL2_MODELS = ("LR", "LR-demo", "B0 item mean", "B2", "B4", "T-b (twin alone)", "B5", "B6", "B7", "B8",
                 "B9 super learner (+T9)")
LEVEL2_LABEL = {"B2": "B2 LR+MF (T1)", "B4": "B4 +T3 (=B3 ngoai R4/R5)", "B5": "B5 +D tho, khong cong",
                "B6": "B6 +D~ (T6)", "B7": "B7 +cong w(z) (T7)", "B8": "B8 T5b (twin thay nen)"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="twin2k", choices=DATASETS)
    ap.add_argument("--regimes", default="R1,R2,R3,R4,R5,R6", help="danh sách chế độ trong R1..R6")
    ap.add_argument("--llm", default=DEFAULT_LLM, help="ollama:<model> | openai:<model> | mock | none")
    ap.add_argument("--llm-cells", type=int, default=30, help="số ô gọi twin T5 mỗi chế độ")
    ap.add_argument("--t5", default="ab", help="biến thể T5: a (độc lập), b (thấy nền), ab")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER, help="st:<model> | ollama:<model> | hash")
    ap.add_argument("--shuffle-texts", action="store_true", help="ablation C3: xáo văn bản câu hỏi trước khi embed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-sparse", type=int, default=80, help="số người thưa ở R4 / người mới ở R5")
    ap.add_argument("--n-known", type=int, default=10, help="số câu đã biết của người thưa ở R4")
    ap.add_argument("--k-cold", type=int, default=5, help="số câu trả lời giữ lại của câu lạnh ở R6")
    ap.add_argument("--lr-engine", default="torch", choices=["torch", "sklearn"],
                    help="LR theo từng câu: torch (mọi bộ dữ liệu, GPU nếu có) hoặc sklearn (tham chiếu, dữ liệu nhỏ)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--max-feat-items", type=int, default=800,
                    help="số câu tối đa làm đặc trưng 'câu trả lời khác' (MovieLens: các phim nhiều đánh giá nhất)")
    ap.add_argument("--gss-year-min", type=int, default=2000)
    ap.add_argument("--gss-max-items", type=int, default=600)
    ap.add_argument("--ml-min-ratings", type=int, default=20)
    ap.add_argument("--max-items", type=int, default=0, help="giới hạn số câu (0 = tất cả), giữ câu nhiều trả lời nhất")
    ap.add_argument("--rank", type=int, default=16, help="hạng MF của T1")
    ap.add_argument("--max-iter", type=int, default=300, help="số vòng L-BFGS của T1")
    ap.add_argument("--max-persons", type=int, default=0, help="lấy mẫu ngẫu nhiên số người (0 = tất cả)")
    ap.add_argument("--bootstrap", type=int, default=300, help="số lần bootstrap theo người")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--quick", action="store_true",
                    help="chạy thử: 600 người, R4 20 người thưa, 8 ô LLM/chế độ, chỉ T5a, 100 bootstrap")
    ap.add_argument("--out", type=Path, default=ROOT / "result")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.max_persons = args.max_persons or 600
        args.n_sparse = min(args.n_sparse, 20)
        args.llm_cells = min(args.llm_cells, 8)
        args.t5 = "a"
        args.bootstrap = min(args.bootstrap, 100)
        args.max_iter = min(args.max_iter, 200)
    return args


# -- formatting ----------------------------------------------------------------------------------------------
def _f(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(point, ci, nd=4):
    if point is None:
        return "-"
    return f"{point:+.{nd}f} [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def table(rows: list[list[str]], header: list[str]) -> tuple[str, str]:
    """(plain aligned text, markdown)"""
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)  # noqa: E731
                               for i, (c, w) in enumerate(zip(r, widths)))
    plain = "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])
    md = "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                   + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
    return plain, md


def metric_rows(evals: dict, models: list[str], label=lambda m: m, with_item=True):
    rows = []
    for m in models:
        if m not in evals:
            continue
        s, d = evals[m]["summary"], evals[m].get("vs_ref")
        r = [label(m), str(s["n"]), _f(s["auc"]), _f(s["auc_item"]) if with_item else None, _f(s["logloss"]),
             _f(s["brier"]), _f(s["acc"], 3),
             "-" if d is None else _ci(d["d_auc"], d["d_auc_ci"]),
             "-" if d is None else _ci(d["d_logloss"], d["d_logloss_ci"])]
        if with_item:
            r.append("-" if d is None or "d_auc_item" not in d else _ci(d["d_auc_item"], d["d_auc_item_ci"]))
        rows.append([c for c in r if c is not None])
    header = ["model", "n", "AUC"] + (["AUC_item"] if with_item else []) + ["logloss", "Brier", "acc",
              "dAUC vs LR [95% CI]", "dLogloss vs LR [95% CI]"] + (["dAUC_item vs LR [95% CI]"] if with_item else [])
    return rows, header


# -- evaluation ----------------------------------------------------------------------------------------------
def evaluate_level1(res: dict, data, B: int, seed: int) -> dict:
    scope = res["scope"]
    y = data.Y[scope].astype(np.float64)
    persons = np.broadcast_to(np.arange(data.shape[0])[:, None], data.shape)[scope]
    # AUC_item = mean AUC over (question x fold): predictions of different folds come from different
    # training sets, and pooling them inside a question would mix fold-level shifts into the ranking
    items = (np.broadcast_to(np.arange(data.shape[1])[None, :], data.shape)[scope].astype(np.int64) * 100
             + res["fold_of"][scope])
    ref = res["preds"][REF][scope]
    out = {}
    for m in LEVEL1_MODELS:
        p = res["preds"][m][scope]
        if np.isnan(p).any():
            continue
        if m == "B4 +T3" and res["t3"]["persons"] == 0:
            continue
        out[m] = {"summary": summary(y, p, items)}
        if m != REF:
            out[m]["vs_ref"] = paired_bootstrap(y, p, ref, persons, items, B=B, seed=seed)
    return out


def evaluate_level2(l2: dict, rows: list[dict], regime: str | None, B: int, seed: int) -> dict:
    idx = np.array([i for i, r in enumerate(rows) if regime is None or r["regime"] == regime])
    if idx.size == 0:
        return {}
    y = np.array([rows[i]["y"] for i in idx])
    persons = np.array([rows[i]["pid"] for i in idx])
    preds = {"LR": np.array([rows[i]["p_LR"] for i in idx]),
             "LR-demo": np.array([rows[i]["p_LR-demo"] for i in idx]),
             "B0 item mean": np.array([rows[i]["p_B0 item mean"] for i in idx])}
    for k, v in l2["pred"].items():
        preds[k] = v[idx]
    out = {}
    for m in LEVEL2_MODELS:
        if m not in preds:
            continue
        out[m] = {"summary": summary(y, preds[m])}
        if m != REF:
            out[m]["vs_ref"] = paired_bootstrap(y, preds[m], preds[REF], persons, None, B=B, seed=seed)
    return out


def twin_tests(l2: dict, rows: list[dict], variants: list[str], regimes: list[str]) -> dict:
    cells = l2["cells"]
    y, eta = cells["y"], cells["eta_B4"]
    reg = np.array([r["regime"] for r in rows])
    out = {"lr_test": {}, "attribution": {}, "diagnostics": {}}
    tests = []
    for rg in regimes + ["all"]:
        mask = np.ones_like(y, dtype=bool) if rg == "all" else reg == rg
        if mask.sum() < 3:
            continue
        for v in variants:
            Dt = l2["Dt"][v][mask]
            t = lr_test(y[mask], eta[mask], Dt)
            out["lr_test"][f"{rg}/T5{v}"] = t
            if rg != "all":
                tests.append((f"{rg}/T5{v}", t["p_value"]))
            D = cells[f"D_{v}"][mask]
            ok = np.isfinite(D)
            var_ratio = float(np.var(Dt[ok]) / np.var(D[ok])) if ok.sum() > 2 and np.var(D[ok]) > 0 else float("nan")
            resid = y[mask] - 1 / (1 + np.exp(-eta[mask]))
            out["diagnostics"][f"{rg}/T5{v}"] = {
                "parse_ok_rate": float(ok.mean()), "var_Dt_over_var_D": var_ratio,
                "corr_Dt_resid": corr_ci(Dt[ok], resid[ok]), "corr_D_eta": corr_ci(D[ok], eta[mask][ok])}
        if "a" in variants:
            ll = lambda p: float(logloss_cells(y[mask], p[mask]).mean())  # noqa: E731
            p_b4 = l2["pred"]["B4"]
            p_b0 = cells["p_B0"]
            after = ll(p_b4) - ll(l2["pred"]["B6"])
            first = ll(p_b0) - ll(l2["pred"]["_B0+twin"])
            out["attribution"][rg] = {"added_after_base_lower": after, "added_first_upper": first,
                                      "shapley_2groups": 0.5 * (after + first), "shared": first - after,
                                      "unit": "giam log-loss trung binh moi o"}
    # Holm over regime x variant
    order = sorted(tests, key=lambda t: t[1])
    m = len(order)
    running = 0.0
    for rank, (name, p) in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p))
        out["lr_test"][name]["p_holm"] = running
    return out


# -- main ----------------------------------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    args = parse_args()
    requested = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in requested:
        if r not in REGIMES:
            sys.exit(f"Che do khong hop le: {r}. Chon trong {sorted(REGIMES)}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / (f"{stamp}_{args.dataset}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"[1/4] Nap {args.dataset} ...")
    kw = {}
    if args.dataset == "gss":
        kw = {"year_min": args.gss_year_min, "max_items": args.gss_max_items}
    elif args.dataset == "movielens":
        kw = {"min_ratings": args.ml_min_ratings}
    data = load_dataset(args.dataset, ROOT, **kw)
    if args.max_items and args.max_items < data.shape[1]:
        top = np.sort(np.argsort(-(~np.isnan(data.Y)).sum(0), kind="stable")[:args.max_items])
        data = data.subset(items=top)
    if args.max_persons and args.max_persons < data.shape[0]:
        keep = np.sort(np.random.default_rng(args.seed).choice(data.shape[0], args.max_persons, replace=False))
        data = data.subset(persons=keep)
    regimes = [r for r in requested if supported(r, data.wave is not None)]
    skipped = [r for r in requested if r not in regimes]
    log(f"      {data.shape[0]} nguoi x {data.shape[1]} cau, {int((~np.isnan(data.Y)).sum())} o co tra loi"
        + (f"; bo du lieu khong co {', '.join(skipped)} (de trong)" if skipped else ""))

    log(f"[2/4] T2 embedding cau hoi ({args.embedder}) ...")
    E = embed_items([it.text() for it in data.items], args.embedder, cache_dir=ROOT / "data/cache",
                    shuffle_texts=args.shuffle_texts, seed=args.seed)

    llm = LLMClient(args.llm, cache_path=ROOT / "data/cache/lrmf_twin_llm.sqlite", seed=0)
    if llm.enabled:
        try:
            llm.check()
        except LLMUnavailable as exc:
            sys.exit(f"LOI LLM: {exc}")
    cfg = RunConfig(regimes=tuple(regimes), n_folds=args.folds, seed=args.seed, n_sparse=args.n_sparse,
                    n_known=args.n_known, k_cold=args.k_cold, llm_cells=args.llm_cells, t5_variants=args.t5,
                    n_jobs=args.jobs, lr_engine=args.lr_engine,
                    base=BaseConfig(rank=args.rank, max_iter=args.max_iter, lam_W=100.0, lam_theta=200.0,
                                    device=args.device, max_feat_items=args.max_feat_items))
    n_t3 = args.n_sparse * sum(r in regimes for r in ("R4", "R5")) if llm.enabled else 0
    n_t5 = len(regimes) * args.llm_cells * len(args.t5) if llm.enabled else 0
    eta = "" if llm.provider != "ollama" else f", uoc tinh ~{(25 * n_t3 + 17 * n_t5) / 60:.0f} phut neu chua co cache (CPU)"
    log(f"      LLM: {args.llm}; toi da {n_t3} loi goi T3 + {n_t5} loi goi T5{eta}")

    log("[3/4] Chay tung che do (moi o duoc du doan ngoai fold) ...")
    results, all_rows, prompts = {}, [], []
    for rg in regimes:
        log(f"  == {rg}: {REGIMES[rg]} ==")
        res = run_regime(data, E, rg, cfg, llm, log=log)
        results[rg] = res
        all_rows += res["level2_rows"]
        prompts += res["prompts"]
    variants = [v for v in args.t5 if v in "ab"] if llm.enabled else []
    l2 = run_level2(all_rows, variants, n_splits=args.folds)

    log("[4/4] Danh gia (bootstrap theo nguoi) ...")
    report, md_parts, metric_records = {}, [], []
    md_parts.append(f"# Twin LR+MF so voi Logistic Regression - {data.source.get('dataset', args.dataset)}\n\nRun `{out_dir.name}`, "
                    f"LLM `{args.llm}`, embedder `{args.embedder}`, {data.shape[0]} nguoi x {data.shape[1]} cau.\n")
    overview = {"dataset": args.dataset, "dataset_name": data.source.get("dataset", args.dataset),
                "shape": list(data.shape), "llm": args.llm, "skipped": skipped, "regimes": {}}
    for rg in skipped:
        md_parts.append(f"## {rg} - {REGIMES[rg]}: khong co tren bo du lieu nay (de trong)\n")
    for rg in regimes:
        res = results[rg]
        ev1 = evaluate_level1(res, data, args.bootstrap, args.seed)
        overview["regimes"][rg] = {"cells": int(res["scope"].sum()), "level1": {
            m: {**{k: e["summary"].get(k) for k in ("auc", "auc_item", "logloss")},
                **{k: v for k, v in (e.get("vs_ref") or {}).items()}} for m, e in ev1.items()}}
        rows, header = metric_rows(ev1, list(LEVEL1_MODELS))
        plain, md = table(rows, header)
        title = f"{rg} - {REGIMES[rg]}: tang 1, moi o trong pham vi ({int(res['scope'].sum())} o)"
        print("\n" + title + "\n" + plain)
        md_parts.append(f"## {title}\n\n{md}\n")
        report[rg] = {"level1": ev1, "folds": res["folds"], "t3": res["t3"], "seconds": res["seconds"]}
        for m, e in ev1.items():
            metric_records.append({"regime": rg, "level": 1, "model": m, **e["summary"],
                                   **{k: v for k, v in (e.get("vs_ref") or {}).items() if not k.endswith("_ci")}})
        if l2 is not None:
            ev2 = evaluate_level2(l2, all_rows, rg, args.bootstrap, args.seed)
            if ev2:
                rows, header = metric_rows(ev2, list(LEVEL2_MODELS), label=lambda m: LEVEL2_LABEL.get(m, m),
                                           with_item=False)
                plain, md = table(rows, header)
                n2 = ev2["LR"]["summary"]["n"]
                title = f"{rg}: tang 2, mau o co goi LLM ({n2} o)"
                print("\n" + title + "\n" + plain)
                md_parts.append(f"## {title}\n\n{md}\n")
                report[rg]["level2"] = ev2
                overview["regimes"][rg]["level2"] = {
                    m: {**{k: e["summary"].get(k) for k in ("n", "auc", "logloss")},
                        **{k: v for k, v in (e.get("vs_ref") or {}).items()}} for m, e in ev2.items()}
                for m, e in ev2.items():
                    metric_records.append({"regime": rg, "level": 2, "model": m, **e["summary"],
                                           **{k: v for k, v in (e.get("vs_ref") or {}).items() if not k.endswith("_ci")}})

    if l2 is not None:
        tests = twin_tests(l2, all_rows, variants, regimes)
        report["twin_tests"] = tests
        overview["lr_test"] = tests["lr_test"]
        overview["attribution"] = tests["attribution"]
        report["super_learner"] = {"candidates": l2["sl_candidates"], "alpha_mean": l2["sl_alpha_mean"]}
        rows = []
        for name, t in tests["lr_test"].items():
            dg = tests["diagnostics"].get(name, {})
            c = dg.get("corr_Dt_resid", {})
            rows.append([name, str(t["n"]), f"{t['beta']:+.3f}", f"{t['chi2']:.2f}", f"{t['p_value']:.4f}",
                         f"{t.get('p_holm', float('nan')):.4f}" if "p_holm" in t else "-",
                         _f(dg.get("var_Dt_over_var_D"), 3),
                         "-" if not c else f"{_f(c.get('r'), 3)} [{_f(c['ci'][0], 2)}, {_f(c['ci'][1], 2)}]",
                         _f(dg.get("parse_ok_rate"), 2)])
        plain, md = table(rows, ["test", "n", "beta", "chi2", "p", "p_Holm", "Var(D~)/Var(D)",
                                 "Corr(D~, Y-p_B4) [95%]", "parse ok"])
        title = "Kiem dinh chinh: LR test cua beta trong logit p = eta_B4 + beta * D~"
        print("\n" + title + "\n" + plain)
        md_parts.append(f"## {title}\n\n{md}\n")
        if tests["attribution"]:
            rows = [[rg, f"{a['added_after_base_lower']:+.4f}", f"{a['added_first_upper']:+.4f}",
                     f"{a['shapley_2groups']:+.4f}", f"{a['shared']:+.4f}"] for rg, a in tests["attribution"].items()]
            plain, md = table(rows, ["regime", "them sau (can duoi)", "them truoc (can tren)", "Shapley", "dung chung"])
            title = "Ghi cong cho twin T5a (giam log-loss moi o)"
            print("\n" + title + "\n" + plain)
            md_parts.append(f"## {title}\n\n{md}\n")
        alpha = ", ".join(f"{c}={a:.3f}" for c, a in zip(l2["sl_candidates"], l2["sl_alpha_mean"]))
        print(f"\nSuper learner T8, trong so alpha trung binh qua fold: {alpha} (khong phai thuoc do dong gop)")
        md_parts.append(f"Super learner T8, alpha trung binh: {alpha} (khong doc alpha nhu dong gop, Muc 5.2).\n")

    if llm.provider == "mock":
        md_parts.append("\n> **Canh bao**: LLM = mock (phan phoi bam tu hash), khong phai mo hinh ngon ngu.\n")
    md_parts.append("\nGhi chu: dAUC/dLogloss = mo hinh - LR; CI 95% bootstrap ghep cap theo nguoi. "
                    "Logloss am hon la tot hon (dLogloss < 0 la thang LR). Tang 2 chi co vai chuc o: CI rong.\n")

    # -- save ------------------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)       # again: an empty folder cleaned up mid-run must not lose it
    (out_dir / "summary.md").write_text("\n".join(md_parts), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in metric_records for k in r}, key=lambda k: (k not in ("regime", "level", "model"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(metric_records)
    (out_dir / "tests.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    overview["llm_stats"] = llm.stats.as_dict()
    (out_dir / "overview.json").write_text(json.dumps(overview, indent=2, default=float), encoding="utf-8")
    if all_rows:
        keys = list(dict.fromkeys(k for r in all_rows for k in r))
        with (out_dir / "level2_cells.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    if prompts:
        with (out_dir / "prompts.jsonl").open("w", encoding="utf-8") as fh:
            for p in prompts:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.save_predictions:
        for rg, res in results.items():
            idx = np.argwhere(res["scope"])
            with gzip.open(out_dir / f"predictions_{rg}.csv.gz", "wt", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["pid", "item_id", "y"] + list(LEVEL1_MODELS))
                for p, j in idx:
                    w.writerow([int(data.pids[p]), data.items[j].item_id, int(data.Y[p, j])]
                               + [f"{res['preds'][m][p, j]:.5f}" for m in LEVEL1_MODELS])
    config = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "run_config": cfg.as_dict(), "lrmf_twin_version": __version__, "data": data.source,
              "llm": llm.describe(), "llm_stats": llm.stats.as_dict(), "python": platform.python_version(),
              "started": stamp, "seconds": round(time.time() - t_all, 1)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"\nDa luu ket qua vao {out_dir}  ({time.time() - t_all:.0f}s; LLM {llm.stats.as_dict()})")
    if llm.stats.failures:
        print(f"CANH BAO: {llm.stats.failures} loi goi LLM that bai (D = NaN o cac o do). Cac loi goi thanh cong "
              f"da duoc cache: chay lai dung lenh nay de chi goi lai cac o loi.")
    if getattr(llm, "cpu_fallback", False):
        print("Ghi chu: runner GPU cua Ollama loi giua chung, phan con lai chay tren CPU (num_gpu=0).")


if __name__ == "__main__":
    main()
