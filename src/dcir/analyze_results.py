from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_METRICS = (
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "subset_accuracy",
    "hamming_loss",
)


def bootstrap_intervals(
    target: np.ndarray,
    prediction: np.ndarray,
    point_metrics: dict,
    iterations: int,
    seed: int,
    chunk_size: int = 25,
) -> pd.DataFrame:
    truth = target.astype(bool)
    predicted = prediction.astype(bool)
    n_examples, n_classes = truth.shape
    tp_unit = (truth & predicted).astype(np.float32)
    fp_unit = (~truth & predicted).astype(np.float32)
    fn_unit = (truth & ~predicted).astype(np.float32)
    support = truth.sum(axis=0).astype(np.int64)
    negative_count = n_examples - support
    observed_tp = tp_unit.sum(axis=0)
    observed_fp = fp_unit.sum(axis=0)
    true_positive_rate = np.divide(
        observed_tp,
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support > 0,
    )
    false_positive_rate = np.divide(
        observed_fp,
        negative_count,
        out=np.zeros(n_classes, dtype=np.float64),
        where=negative_count > 0,
    )
    exact_unit = np.all(truth == predicted, axis=1).astype(np.float32)
    errors_unit = np.not_equal(truth, predicted).sum(axis=1).astype(np.float32)
    probability = np.full(n_examples, 1.0 / n_examples)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in BOOTSTRAP_METRICS}

    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        weights = rng.multinomial(
            n_examples, probability, size=count
        ).astype(np.float32)
        # Preserve each label's positive/negative support for macro metrics.
        # Ordinary row bootstrap can omit very rare labels entirely and then
        # count them as F1=0, producing a strongly downward-biased macro CI.
        tp = rng.binomial(
            support, true_positive_rate, size=(count, n_classes)
        ).astype(np.float64)
        fp = rng.binomial(
            negative_count, false_positive_rate, size=(count, n_classes)
        ).astype(np.float64)
        fn = support[None, :] - tp
        class_precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
        )
        class_recall = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        class_f1 = np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros_like(tp),
            where=(2 * tp + fp + fn) > 0,
        )
        sample_tp = weights @ tp_unit
        sample_fp = weights @ fp_unit
        sample_fn = weights @ fn_unit
        total_tp = sample_tp.sum(axis=1)
        total_fp = sample_fp.sum(axis=1)
        total_fn = sample_fn.sum(axis=1)
        samples["macro_f1"].extend(class_f1.mean(axis=1))
        samples["micro_f1"].extend(
            np.divide(
                2 * total_tp,
                2 * total_tp + total_fp + total_fn,
                out=np.zeros_like(total_tp),
                where=(2 * total_tp + total_fp + total_fn) > 0,
            )
        )
        samples["macro_precision"].extend(class_precision.mean(axis=1))
        samples["macro_recall"].extend(class_recall.mean(axis=1))
        samples["subset_accuracy"].extend(
            (weights @ exact_unit) / n_examples
        )
        samples["hamming_loss"].extend(
            (weights @ errors_unit) / (n_examples * n_classes)
        )

    rows = []
    for name in BOOTSTRAP_METRICS:
        values = np.asarray(samples[name], dtype=np.float64)
        method = (
            "label_stratified"
            if name in {"macro_f1", "macro_precision", "macro_recall"}
            else "sample"
        )
        rows.append(
            {
                "metric": name,
                "point_estimate": float(point_metrics[name]),
                "ci_2.5%": float(np.quantile(values, 0.025)),
                "ci_97.5%": float(np.quantile(values, 0.975)),
                "bootstrap_method": method,
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def sample_error_table(
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    truth = target.astype(bool)
    predicted = prediction.astype(bool)
    tp = (truth & predicted).sum(axis=1)
    fp = (~truth & predicted).sum(axis=1)
    fn = (truth & ~predicted).sum(axis=1)
    denominator = 2 * tp + fp + fn
    sample_f1 = np.divide(
        2 * tp,
        denominator,
        out=np.ones_like(tp, dtype=np.float64),
        where=denominator > 0,
    )

    rows = []
    for index in np.where((fp + fn) > 0)[0]:
        false_positive = np.where(~truth[index] & predicted[index])[0]
        false_negative = np.where(truth[index] & ~predicted[index])[0]
        rows.append(
            {
                "sample_index": int(index),
                "d1": samples.iloc[index]["d1"],
                "d2": samples.iloc[index]["d2"],
                "record_ids": samples.iloc[index]["record_ids"],
                "true_labels": ";".join(
                    str(value + 1) for value in np.where(truth[index])[0]
                ),
                "predicted_labels": ";".join(
                    str(value + 1) for value in np.where(predicted[index])[0]
                ),
                "false_positive_labels": ";".join(
                    str(value + 1) for value in false_positive
                ),
                "false_negative_labels": ";".join(
                    str(value + 1) for value in false_negative
                ),
                "false_positive_max_probability": (
                    float(probability[index, false_positive].max())
                    if len(false_positive)
                    else None
                ),
                "false_negative_max_probability": (
                    float(probability[index, false_negative].max())
                    if len(false_negative)
                    else None
                ),
                "tp": int(tp[index]),
                "fp": int(fp[index]),
                "fn": int(fn[index]),
                "sample_f1": float(sample_f1[index]),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["sample_f1", "fn", "fp"],
            ascending=[True, False, False],
        )
    return result


def make_plots(
    metrics: dict,
    classes: pd.DataFrame,
    intervals: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    generated = []
    overall_names = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_auprc_supported",
        "macro_auroc_supported",
        "subset_accuracy",
    ]
    labels = [
        "Macro-F1",
        "Micro-F1",
        "Macro-P",
        "Macro-R",
        "Macro-AUPRC",
        "Macro-AUROC",
        "Subset Acc.",
    ]
    values = [float(metrics[name]) for name in overall_names]
    width, height = 940, 500
    plot_left, plot_top, plot_width, plot_height = 70, 55, 830, 350
    bar_width = plot_width / len(values) * 0.62
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="470" y="30" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">DeepDDI test metrics</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" '
        f'x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" '
        'stroke="#333"/>',
    ]
    for tick in range(6):
        score = tick / 5
        y = plot_top + plot_height * (1 - score)
        elements.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width}" y2="{y:.1f}" '
            'stroke="#ddd"/>'
        )
        elements.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-family="sans-serif" '
            f'font-size="12">{score:.1f}</text>'
        )
    slot = plot_width / len(values)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = plot_left + index * slot + (slot - bar_width) / 2
        y = plot_top + plot_height * (1 - value)
        h = plot_height * value
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{h:.1f}" fill="#3973ac"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="12">{value:.3f}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="430" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="11">{escape(label)}</text>',
            ]
        )
    elements.append("</svg>")
    (output_dir / "overall_metrics.svg").write_text(
        "\n".join(elements), encoding="utf-8"
    )
    generated.append("overall_metrics.svg")

    ranked = classes.sort_values("f1")
    row_height = 26
    class_height = 75 + len(ranked) * row_height
    class_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" '
        f'height="{class_height}" viewBox="0 0 900 {class_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">Per-class F1 '
        '(ascending)</text>',
    ]
    for row_index, row in enumerate(ranked.itertuples(index=False)):
        y = 50 + row_index * row_height
        value = float(row.f1)
        color = "#3a9d5d" if value >= 0.8 else "#d47b32"
        class_elements.extend(
            [
                f'<text x="55" y="{y + 14}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">'
                f'{escape(str(row.class_raw))}</text>',
                f'<rect x="65" y="{y}" width="{760 * value:.1f}" '
                f'height="17" fill="{color}"/>',
                f'<text x="{75 + 760 * value:.1f}" y="{y + 14}" '
                f'font-family="sans-serif" font-size="11">'
                f'{value:.3f} (n={int(row.support)})</text>',
            ]
        )
    class_elements.append("</svg>")
    (output_dir / "per_class_f1.svg").write_text(
        "\n".join(class_elements), encoding="utf-8"
    )
    generated.append("per_class_f1.svg")

    scatter_width, scatter_height = 820, 600
    left, top, chart_width, chart_height = 75, 55, 680, 470
    support = classes["support"].to_numpy(dtype=float)
    log_support = np.log10(np.maximum(support, 1))
    log_min, log_max = float(log_support.min()), float(log_support.max())
    log_span = max(log_max - log_min, 1e-8)
    scatter_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{scatter_width}" '
        f'height="{scatter_height}" viewBox="0 0 {scatter_width} '
        f'{scatter_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">Class support versus F1'
        "</text>",
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + chart_height}" stroke="#333"/>',
        '<text x="410" y="580" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Test support (log scale)</text>',
        '<text x="18" y="290" text-anchor="middle" '
        'transform="rotate(-90 18 290)" font-family="sans-serif" '
        'font-size="13">F1</text>',
    ]
    for row_index, row in enumerate(classes.itertuples(index=False)):
        x = left + (
            (log_support[row_index] - log_min) / log_span
        ) * chart_width
        y = top + (1 - float(row.f1)) * chart_height
        recall = float(row.recall)
        red = int(210 * (1 - recall) + 40 * recall)
        green = int(95 * (1 - recall) + 145 * recall)
        blue = int(45 * (1 - recall) + 95 * recall)
        scatter_elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
            f'fill="rgb({red},{green},{blue})" fill-opacity="0.8">'
            f'<title>class={escape(str(row.class_raw))}, '
            f'support={int(row.support)}, F1={float(row.f1):.4f}, '
            f'recall={recall:.4f}</title></circle>'
        )
    scatter_elements.append("</svg>")
    (output_dir / "support_vs_f1.svg").write_text(
        "\n".join(scatter_elements), encoding="utf-8"
    )
    generated.append("support_vs_f1.svg")

    f1_intervals = intervals[
        intervals["metric"].isin(["macro_f1", "micro_f1"])
    ].copy()
    ci_elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="420" '
        'viewBox="0 0 650 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="325" y="30" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">F1 bootstrap 95% CI</text>',
    ]
    for index, (_, row) in enumerate(f1_intervals.iterrows()):
        y = 130 + index * 130
        lower = float(row["ci_2.5%"])
        upper = float(row["ci_97.5%"])
        point = float(row["point_estimate"])
        x_low = 75 + lower * 500
        x_high = 75 + upper * 500
        x_point = 75 + point * 500
        ci_elements.extend(
            [
                f'<text x="65" y="{y + 5}" text-anchor="end" '
                f'font-family="sans-serif" font-size="13">'
                f'{escape(str(row["metric"]))}</text>',
                f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" '
                f'y2="{y}" stroke="#713e8a" stroke-width="4"/>',
                f'<circle cx="{x_point:.1f}" cy="{y}" r="7" '
                'fill="#713e8a"/>',
                f'<text x="{x_point:.1f}" y="{y - 15}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">'
                f'{point:.4f} [{lower:.4f}, {upper:.4f}]</text>',
            ]
        )
    ci_elements.append("</svg>")
    (output_dir / "f1_bootstrap_ci.svg").write_text(
        "\n".join(ci_elements), encoding="utf-8"
    )
    generated.append("f1_bootstrap_ci.svg")
    return generated


def analyze(
    result_dir: Path,
    output_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    with (result_dir / "metrics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metrics = json.load(handle)
    classes = pd.read_csv(
        result_dir / "per_class_metrics.csv",
        dtype={"class_raw": str},
    )
    samples = pd.read_csv(
        result_dir / "samples.csv", dtype={"record_ids": str}
    )
    with np.load(result_dir / "predictions.npz", allow_pickle=False) as data:
        target = np.asarray(data["target"])
        probability = np.asarray(data["probability"])
        prediction = np.asarray(data["prediction"])

    if metrics["task_mode"] != "multilabel":
        raise ValueError("This analysis currently expects multilabel results.")
    if not len(target) == len(probability) == len(prediction) == len(samples):
        raise RuntimeError("Prediction and sample counts do not match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = classes.sort_values(
        ["f1", "support"], ascending=[False, False]
    )
    ranked.to_csv(output_dir / "class_metrics_ranked.csv", index=False)
    ranked.head(10).to_csv(output_dir / "best_10_classes.csv", index=False)
    ranked.tail(10).sort_values("f1").to_csv(
        output_dir / "worst_10_classes.csv", index=False
    )
    classes.sort_values(["support", "f1"]).head(10).to_csv(
        output_dir / "rarest_10_classes.csv", index=False
    )

    errors = sample_error_table(
        target, prediction, probability, samples
    )
    errors.to_csv(output_dir / "error_cases.csv", index=False)
    errors.head(100).to_csv(
        output_dir / "hardest_100_cases.csv", index=False
    )
    intervals = bootstrap_intervals(
        target,
        prediction,
        metrics,
        bootstrap_iterations,
        bootstrap_seed,
    )
    intervals.to_csv(output_dir / "bootstrap_95ci.csv", index=False)

    validation_f1 = float(
        metrics.get("checkpoint_validation", {}).get("macro_f1", np.nan)
    )
    summary = {
        "dataset": metrics["dataset"],
        "checkpoint_epoch": metrics.get("checkpoint_epoch"),
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "test_examples": int(metrics["examples"]),
        "classes": int(metrics["classes"]),
        "threshold": float(metrics["threshold"]),
        "validation_macro_f1": validation_f1,
        "test_macro_f1": float(metrics["macro_f1"]),
        "validation_test_macro_f1_gap": (
            float(validation_f1 - metrics["macro_f1"])
            if np.isfinite(validation_f1)
            else None
        ),
        "exact_match_examples": int(
            np.all(target == prediction, axis=1).sum()
        ),
        "error_examples": int(len(errors)),
        "perfect_f1_classes": int((classes["f1"] == 1.0).sum()),
        "classes_f1_below_0.8": int((classes["f1"] < 0.8).sum()),
        "classes_f1_below_0.5": int((classes["f1"] < 0.5).sum()),
        "median_class_f1": float(classes["f1"].median()),
        "median_class_support": float(classes["support"].median()),
        "metrics": metrics,
    }
    with (output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    generated_plots = make_plots(
        metrics, classes, intervals, output_dir
    )
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    ci_lookup = intervals.set_index("metric")
    macro_ci = ci_lookup.loc["macro_f1"]
    report = f"""# DeepDDI 单次测试结果分析

## 实验设置

- 数据集：{metrics['dataset']}
- 任务：{metrics['task_mode']}
- 划分：{metrics['split']}（随机药物对 8:1:1）
- 测试样本：{metrics['examples']:,}
- 类别数：{metrics['classes']}
- 固定阈值：{metrics['threshold']}
- 最佳 checkpoint epoch：{metrics.get('checkpoint_epoch')}
- checkpoint SHA-256：`{metrics.get('checkpoint_sha256')}`

## 总体结果

- Macro-F1：{metrics['macro_f1']:.6f}（Bootstrap 95% CI：
  {macro_ci['ci_2.5%']:.6f}–{macro_ci['ci_97.5%']:.6f}）
- Micro-F1：{metrics['micro_f1']:.6f}
- Weighted-F1：{metrics['weighted_f1']:.6f}
- Macro-Precision：{metrics['macro_precision']:.6f}
- Macro-Recall：{metrics['macro_recall']:.6f}
- Macro-AUPRC：{metrics['macro_auprc_supported']:.6f}
- Macro-AUROC：{metrics['macro_auroc_supported']:.6f}
- Subset Accuracy：{metrics['subset_accuracy']:.6f}
- Hamming Loss：{metrics['hamming_loss']:.6f}
- 验证—测试 Macro-F1 差值：{summary['validation_test_macro_f1_gap']:.6f}

## 类别与错误分析

- 类别 F1 中位数：{summary['median_class_f1']:.6f}
- F1 < 0.8 的类别数：{summary['classes_f1_below_0.8']}
- F1 < 0.5 的类别数：{summary['classes_f1_below_0.5']}
- 最佳类别：{best['class_raw']}（F1={best['f1']:.6f}，
  support={int(best['support'])}）
- 最弱类别：{worst['class_raw']}（F1={worst['f1']:.6f}，
  support={int(worst['support'])}）
- 完全正确药物对：{summary['exact_match_examples']:,}
- 至少一个标签错误的药物对：{summary['error_examples']:,}

## 结果文件

- `class_metrics_ranked.csv`：全部类别排序
- `best_10_classes.csv` / `worst_10_classes.csv`
- `rarest_10_classes.csv`
- `error_cases.csv` / `hardest_100_cases.csv`
- `bootstrap_95ci.csv`
- `analysis_summary.json`
- 图表：{', '.join(generated_plots) if generated_plots else '未生成（缺少 matplotlib）'}

该报告只分析已冻结的测试预测，没有重新训练、重新推理或调整阈值。
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"All analysis outputs written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze frozen DeepDDI test predictions."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    analyze(
        args.result_dir.resolve(),
        args.output_dir.resolve(),
        args.bootstrap,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
