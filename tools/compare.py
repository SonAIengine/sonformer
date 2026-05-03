"""
실험 결과 비교 표 출력.

각 experiments/<NN-name>/results/eval.json + model_meta.json + config.json을 모아
한 표로 출력. 학습 안 된 폴더는 자동 스킵.

Usage:
    python tools/compare.py
    python tools/compare.py --csv results_summary.csv     # CSV로 저장
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments"


def load_experiment(folder: Path) -> dict | None:
    """폴더에서 실험 메타+결과 로드. 학습 안 됐으면 None."""
    eval_path = folder / "results" / "eval.json"
    if not eval_path.exists():
        return None

    eval_data = json.loads(eval_path.read_text())
    config_path = folder / "results" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    final = eval_data.get("final", {})
    n_params = eval_data.get("n_params", 0)

    row = {
        "name": folder.name,
        "dataset": config.get("data", {}).get("dataset", "?"),
        "params_M": n_params / 1e6 if n_params else None,
        "eval_loss": final.get("loss"),
        "eval_ppl": final.get("ppl"),
        "best_eval_loss": eval_data.get("best_eval_loss"),
        "wallclock_s": eval_data.get("wallclock_s"),
        "total_steps": eval_data.get("total_steps"),
    }

    # extrapolation (RoPE 류 ablation에 있음)
    extrap = eval_data.get("extrapolation", {})
    if extrap:
        row["extrap"] = {k: v.get("ppl") for k, v in extrap.items()}

    return row


def fmt(v, kind="num"):
    if v is None:
        return "—"
    if isinstance(v, float):
        if kind == "ppl":
            return f"{v:.2f}"
        if kind == "loss":
            return f"{v:.4f}"
        if kind == "params":
            return f"{v:.2f}M"
        if kind == "time":
            return f"{v:.1f}s"
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="CSV 출력 경로")
    args = ap.parse_args()

    folders = sorted(p for p in EXP_DIR.iterdir() if p.is_dir() and (p / "config.yaml").exists())
    rows = []
    for folder in folders:
        row = load_experiment(folder)
        if row is None:
            print(f"  (skip {folder.name}: not yet run)")
            continue
        rows.append(row)

    if not rows:
        print("\nNo completed experiments found.")
        return

    # baseline을 찾아서 Δ 컬럼 채움
    baseline_ppl = None
    baseline_loss = None
    for r in rows:
        if r["name"] == "00-baseline":
            baseline_ppl = r["eval_ppl"]
            baseline_loss = r["eval_loss"]
            break

    print()
    print(f"{'experiment':<20} {'dataset':<14} {'params':<8} {'eval_loss':<10} {'eval_ppl':<10} {'Δ ppl':<10} {'wallclock':<10} {'steps':<8}")
    print("-" * 100)
    for r in rows:
        delta = ""
        if baseline_ppl and r["name"] != "00-baseline" and r["eval_ppl"]:
            d = (r["eval_ppl"] - baseline_ppl) / baseline_ppl * 100
            delta = f"{d:+.1f}%"
        print(
            f"{r['name']:<20} "
            f"{r['dataset']:<14} "
            f"{fmt(r['params_M'], 'params'):<8} "
            f"{fmt(r['eval_loss'], 'loss'):<10} "
            f"{fmt(r['eval_ppl'], 'ppl'):<10} "
            f"{delta:<10} "
            f"{fmt(r['wallclock_s'], 'time'):<10} "
            f"{r['total_steps'] or '—':<8}"
        )

    # extrapolation 결과가 있는 실험들
    print()
    extrap_rows = [r for r in rows if "extrap" in r]
    if extrap_rows:
        print("Extrapolation PPL (sequence length):")
        all_lens = sorted({int(L) for r in extrap_rows for L in r["extrap"].keys()})
        header = f"{'experiment':<20} " + " ".join(f"T={L:<6}" for L in all_lens)
        print(header)
        print("-" * len(header))
        for r in extrap_rows:
            cells = []
            for L in all_lens:
                v = r["extrap"].get(str(L))
                cells.append(f"{v:.1f}" if isinstance(v, (int, float)) else "  —  ")
            print(f"{r['name']:<20} " + " ".join(f"{c:<8}" for c in cells))

    # CSV
    if args.csv:
        out = Path(args.csv)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "dataset", "params_M", "eval_loss", "eval_ppl", "best_eval_loss", "wallclock_s", "total_steps"])
            for r in rows:
                w.writerow([r["name"], r["dataset"], r["params_M"], r["eval_loss"], r["eval_ppl"], r["best_eval_loss"], r["wallclock_s"], r["total_steps"]])
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
