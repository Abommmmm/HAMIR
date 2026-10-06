from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MACRO_METRICS = ("macro_f1", "macro_precision", "macro_recall")
SAMPLE_METRICS = (
    "micro_f1",
    "subset_accuracy",
    "hamming_loss",
)


def _rates(tp, fp, fn):
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0,
    )
    return precision, recall, f1


def point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    tp = (target & prediction).sum(axis=0)
    fp = (~target & prediction).sum(axis=0)
    fn = (target & ~prediction).sum(axis=0)
    precision, recall, f1 = _rates(tp, fp, fn)
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_f1 = np.divide(
        2 * total_tp,
        2 * total_tp + total_fp + total_fn,
    )
    return {
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "micro_f1": float(micro_f1),
        "subset_accuracy": float(np.all(target == prediction, axis=1).mean()),
        "hamming_loss": float(np.not_equal(target, prediction).mean()),
    }


def _paired_outcome_probabilities(
    prediction0: np.ndarray,
    prediction1: np.ndarray,
) -> np.ndarray:
    category = prediction0.astype(np.int8) * 2 + prediction1.astype(np.int8)
    counts = np.bincount(category, minlength=4)
    total = counts.sum()
    if total == 0:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return counts / total


def macro_bootstrap(
    target: np.ndarray,
    prediction0: np.ndarray,
    prediction1: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n_classes = target.shape[1]
    tp0 = np.zeros((iterations, n_classes), dtype=np.float64)
    tp1 = np.zeros_like(tp0)
    fp0 = np.zeros_like(tp0)
    fp1 = np.zeros_like(tp0)
    support = target.sum(axis=0).astype(np.int64)
    negative = len(target) - support

    for label in range(n_classes):
        positive_mask = target[:, label]
        positive_draw = rng.multinomial(
            int(support[label]),
            _paired_outcome_probabilities(
                prediction0[positive_mask, label],
                prediction1[positive_mask, label],
            ),
            size=iterations,
        )
        negative_draw = rng.multinomial(
            int(negative[label]),
            _paired_outcome_probabilities(
                prediction0[~positive_mask, label],
                prediction1[~positive_mask, label],
            ),
            size=iterations,
        )
        tp0[:, label] = positive_draw[:, 2] + positive_draw[:, 3]
        tp1[:, label] = positive_draw[:, 1] + positive_draw[:, 3]
        fp0[:, label] = negative_draw[:, 2] + negative_draw[:, 3]
        fp1[:, label] = negative_draw[:, 1] + negative_draw[:, 3]

    fn0 = support[None, :] - tp0
    fn1 = support[None, :] - tp1
    precision0, recall0, f10 = _rates(tp0, fp0, fn0)
    precision1, recall1, f11 = _rates(tp1, fp1, fn1)
    return {
        "macro_f1": (f10.mean(axis=1), f11.mean(axis=1)),
        "macro_precision": (
            precision0.mean(axis=1),
            precision1.mean(axis=1),
        ),
        "macro_recall": (recall0.mean(axis=1), recall1.mean(axis=1)),
    }


def sample_bootstrap(
    target: np.ndarray,
    prediction0: np.ndarray,
    prediction1: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n_examples, n_classes = target.shape
    probability = np.full(n_examples, 1.0 / n_examples)
    units = []
    for prediction in (prediction0, prediction1):
        units.append(
            {
                "tp": (target & prediction).sum(axis=1).astype(np.float32),
                "fp": (~target & prediction).sum(axis=1).astype(np.float32),
                "fn": (target & ~prediction).sum(axis=1).astype(np.float32),
                "exact": np.all(target == prediction, axis=1).astype(
                    np.float32
                ),
                "errors": np.not_equal(target, prediction).sum(
                    axis=1
                ).astype(np.float32),
            }
        )
    samples = {
        name: ([], []) for name in SAMPLE_METRICS
    }
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        weights = rng.multinomial(
            n_examples, probability, size=count
        ).astype(np.float32)
        for model_index, unit in enumerate(units):
            tp = weights @ unit["tp"]
            fp = weights @ unit["fp"]
            fn = weights @ unit["fn"]
            micro_f1 = np.divide(
                2 * tp,
                2 * tp + fp + fn,
                out=np.zeros_like(tp),
                where=(2 * tp + fp + fn) > 0,
            )
            samples["micro_f1"][model_index].extend(micro_f1)
            samples["subset_accuracy"][model_index].extend(
                (weights @ unit["exact"]) / n_examples
            )
            samples["hamming_loss"][model_index].extend(
                (weights @ unit["errors"]) / (n_examples * n_classes)
            )
    return {
        name: (
            np.asarray(values[0], dtype=np.float64),
            np.asarray(values[1], dtype=np.float64),
        )
        for name, values in samples.items()
    }


def summarize_bootstrap(
    distributions: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics0: dict,
    metrics1: dict,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for metric, (values0, values1) in distributions.items():
        delta = values1 - values0
        nonpositive = (np.count_nonzero(delta <= 0) + 1) / (iterations + 1)
        nonnegative = (np.count_nonzero(delta >= 0) + 1) / (iterations + 1)
        rows.append(
            {
                "metric": metric,
                "m0": float(metrics0[metric]),
                "m1": float(metrics1[metric]),
                "delta_m1_minus_m0": float(
                    metrics1[metric] - metrics0[metric]
                ),
                "delta_ci_2.5%": float(np.quantile(delta, 0.025)),
                "delta_ci_97.5%": float(np.quantile(delta, 0.975)),
                "probability_delta_gt_0": float(np.mean(delta > 0)),
                "paired_bootstrap_p_two_sided": float(
                    min(1.0, 2 * min(nonpositive, nonnegative))
                ),
                "iterations": iterations,
                "seed": seed,
                "method": (
                    "paired_label_stratified"
                    if metric in MACRO_METRICS
                    else "paired_sample"
                ),
            }
        )
    return pd.DataFrame(rows)


def per_class_deltas(
    target: np.ndarray,
    prediction0: np.ndarray,
    prediction1: np.ndarray,
    class_raw: pd.Series,
) -> pd.DataFrame:
    tp0 = (target & prediction0).sum(axis=0)
    fp0 = (~target & prediction0).sum(axis=0)
    fn0 = (target & ~prediction0).sum(axis=0)
    tp1 = (target & prediction1).sum(axis=0)
    fp1 = (~target & prediction1).sum(axis=0)
    fn1 = (target & ~prediction1).sum(axis=0)
    precision0, recall0, f10 = _rates(tp0, fp0, fn0)
    precision1, recall1, f11 = _rates(tp1, fp1, fn1)
    return pd.DataFrame(
        {
            "class_internal": np.arange(target.shape[1]),
            "class_raw": class_raw.astype(str),
            "support": target.sum(axis=0),
            "m0_precision": precision0,
            "m1_precision": precision1,
            "delta_precision": precision1 - precision0,
            "m0_recall": recall0,
            "m1_recall": recall1,
            "delta_recall": recall1 - recall0,
            "m0_f1": f10,
            "m1_f1": f11,
            "delta_f1": f11 - f10,
        }
    ).sort_values("delta_f1", ascending=False)


def write_delta_svg(frame: pd.DataFrame, path: Path) -> None:
    width, height = 840, 430
    left, top, chart_width, row_height = 230, 55, 540, 62
    max_abs = max(
        float(
            np.max(
                np.abs(
                    frame[
                        ["delta_ci_2.5%", "delta_ci_97.5%"]
                    ].to_numpy()
                )
            )
        ),
        1e-8,
    )
    origin = left + chart_width / 2
    half = chart_width / 2
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="420" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">'
        "Paired bootstrap: M1 - M0</text>",
        f'<line x1="{origin}" y1="42" x2="{origin}" y2="{height - 30}" '
        'stroke="#555"/>',
    ]
    for index, row in frame.iterrows():
        y = top + index * row_height
        lower = origin + float(row["delta_ci_2.5%"]) / max_abs * half
        upper = origin + float(row["delta_ci_97.5%"]) / max_abs * half
        point = origin + float(row["delta_m1_minus_m0"]) / max_abs * half
        elements.extend(
            [
                f'<text x="{left - 15}" y="{y + 5}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{row["metric"]}</text>',
                f'<line x1="{lower:.1f}" y1="{y}" x2="{upper:.1f}" '
                f'y2="{y}" stroke="#3973ac" stroke-width="4"/>',
                f'<circle cx="{point:.1f}" cy="{y}" r="6" fill="#3973ac"/>',
                f'<text x="{upper + 8:.1f}" y="{y + 5}" '
                f'font-family="sans-serif" font-size="11">'
                f'{float(row["delta_m1_minus_m0"]):+.5f}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def analyze(
    m0_dir: Path,
    m1_dir: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> None:
    with np.load(m0_dir / "predictions.npz", allow_pickle=False) as archive:
        target0 = np.asarray(archive["target"]).astype(bool)
        prediction0 = np.asarray(archive["prediction"]).astype(bool)
    with np.load(m1_dir / "predictions.npz", allow_pickle=False) as archive:
        target1 = np.asarray(archive["target"]).astype(bool)
        prediction1 = np.asarray(archive["prediction"]).astype(bool)
    if not np.array_equal(target0, target1):
        raise RuntimeError("M0 and M1 targets or sample order do not match.")
    if prediction0.shape != prediction1.shape:
        raise RuntimeError("M0 and M1 prediction shapes do not match.")
    metrics0 = point_metrics(target0, prediction0)
    metrics1 = point_metrics(target1, prediction1)
    class_path = m0_dir / "per_class_metrics.csv"
    if class_path.exists():
        classes0 = pd.read_csv(class_path, dtype={"class_raw": str})
        class_raw = classes0["class_raw"]
    else:
        class_raw = pd.Series(np.arange(target0.shape[1]))

    rng = np.random.default_rng(seed)
    distributions = macro_bootstrap(
        target0, prediction0, prediction1, iterations, rng
    )
    distributions.update(
        sample_bootstrap(
            target0, prediction0, prediction1, iterations, rng
        )
    )
    summary = summarize_bootstrap(
        distributions, metrics0, metrics1, iterations, seed
    )
    class_delta = per_class_deltas(
        target0, prediction0, prediction1, class_raw
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    class_delta.to_csv(output_dir / "per_class_delta.csv", index=False)
    write_delta_svg(summary, output_dir / "paired_bootstrap_delta.svg")

    macro = summary.loc[summary["metric"] == "macro_f1"].iloc[0]
    improved = int((class_delta["delta_f1"] > 0).sum())
    worsened = int((class_delta["delta_f1"] < 0).sum())
    unchanged = int((class_delta["delta_f1"] == 0).sum())
    report = f"""# M1 versus M0 paired Bootstrap

- Test examples: {len(target0):,}
- Classes: {target0.shape[1]}
- Iterations: {iterations:,}
- Seed: {seed}
- M0 Macro-F1: {macro['m0']:.6f}
- M1 Macro-F1: {macro['m1']:.6f}
- Paired delta: {macro['delta_m1_minus_m0']:+.6f}
- Delta 95% CI: [{macro['delta_ci_2.5%']:+.6f},
  {macro['delta_ci_97.5%']:+.6f}]
- P(delta > 0): {macro['probability_delta_gt_0']:.6f}
- Paired bootstrap two-sided p: {macro['paired_bootstrap_p_two_sided']:.6f}
- Per-class F1 improved/worsened/unchanged:
  {improved}/{worsened}/{unchanged}

Macro metrics use paired label-stratified resampling. Micro-F1, subset
accuracy, and Hamming loss use paired sample resampling. Both models receive
the same resampled observations in every replicate.
"""
    (output_dir / "paired_bootstrap_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print(f"Paired results written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired Bootstrap comparison of frozen M0 and M1 results."
    )
    parser.add_argument("--m0-dir", required=True, type=Path)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("--iterations must be at least 1000.")
    analyze(
        args.m0_dir.resolve(),
        args.m1_dir.resolve(),
        args.output_dir.resolve(),
        args.iterations,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
