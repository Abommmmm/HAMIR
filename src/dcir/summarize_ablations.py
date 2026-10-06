from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


VARIANTS = {
    "M0": "PaiNN baseline",
    "M1": "PaiNN + DCIA",
    "M2": "PaiNN + CRDM",
    "M3": "PaiNN + DCIA + CRDM",
}


def write_svg(frame: pd.DataFrame, path: Path) -> None:
    width, height = 900, 470
    left, top, chart_width, chart_height = 80, 55, 760, 320
    metrics = ("macro_f1", "micro_f1")
    colors = {"macro_f1": "#3973ac", "micro_f1": "#3a9d5d"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">'
        "DeepDDI architecture ablation</text>",
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_height * (1 - value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" '
                f'x2="{left + chart_width}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.1f}</text>',
            ]
        )
    group_width = chart_width / len(frame)
    bar_width = group_width * 0.28
    for group_index, row in frame.reset_index(drop=True).iterrows():
        center = left + group_width * (group_index + 0.5)
        for metric_index, metric in enumerate(metrics):
            value = float(row[metric])
            x = center + (metric_index - 1) * bar_width
            y = top + chart_height * (1 - value)
            elements.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{chart_height * value:.1f}" '
                    f'fill="{colors[metric]}"/>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11">{value:.3f}</text>',
                ]
            )
        elements.append(
            f'<text x="{center:.1f}" y="400" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{row["variant"]}</text>'
        )
    elements.extend(
        [
            '<rect x="300" y="435" width="14" height="14" fill="#3973ac"/>',
            '<text x="322" y="447" font-family="sans-serif" '
            'font-size="12">Macro-F1</text>',
            '<rect x="450" y="435" width="14" height="14" fill="#3a9d5d"/>',
            '<text x="472" y="447" font-family="sans-serif" '
            'font-size="12">Micro-F1</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def summarize(
    checkpoints_root: Path,
    results_root: Path,
    output_dir: Path,
) -> None:
    rows = []
    for variant, description in VARIANTS.items():
        key = variant.lower()
        checkpoint_path = checkpoints_root / key / "best.pt"
        metrics_path = results_root / key / "metrics.json"
        history_path = checkpoints_root / key / "history.csv"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        rows.append(
            {
                "variant": variant,
                "description": description,
                "best_epoch": int(checkpoint["epoch"]),
                "trainable_parameters": int(
                    checkpoint["trainable_parameter_count"]
                ),
                "training_hours": float(
                    checkpoint["elapsed_seconds"] / 3600
                ),
                "completed_epochs": int(len(history)),
                "validation_macro_f1": float(
                    checkpoint["validation"]["macro_f1"]
                ),
                "macro_f1": float(metrics["macro_f1"]),
                "micro_f1": float(metrics["micro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_auprc": float(metrics["macro_auprc_supported"]),
                "macro_auroc": float(metrics["macro_auroc_supported"]),
                "subset_accuracy": float(metrics["subset_accuracy"]),
                "hamming_loss": float(metrics["hamming_loss"]),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = float(frame.loc[frame["variant"] == "M0", "macro_f1"].iloc[0])
    full = float(frame.loc[frame["variant"] == "M3", "macro_f1"].iloc[0])
    frame["macro_f1_delta_vs_m0"] = frame["macro_f1"] - baseline
    frame["macro_f1_delta_vs_m3"] = frame["macro_f1"] - full
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "ablation_summary.csv", index=False)
    write_svg(frame, output_dir / "ablation_macro_micro_f1.svg")

    best = frame.sort_values("macro_f1", ascending=False).iloc[0]
    table_lines = [
        "| Variant | Parameters | Best epoch | Macro-F1 | Micro-F1 | "
        "Macro-AUPRC | Δ Macro-F1 vs M0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        table_lines.append(
            f"| {row.variant} | {row.trainable_parameters:,} | "
            f"{row.best_epoch} | {row.macro_f1:.6f} | "
            f"{row.micro_f1:.6f} | {row.macro_auprc:.6f} | "
            f"{row.macro_f1_delta_vs_m0:+.6f} |"
        )
    table = "\n".join(table_lines)
    report = f"""# DeepDDI 架构消融实验

## 公平性协议

- 相同 DeepDDI 8:1:1 grouped-pair split
- 相同 seed 17、类别权重上限和固定测试阈值 0.5
- 所有 M0–M3 均从头训练
- Epoch 1–10 学习率 2e-4，Epoch 11 后学习率 5e-5
- 最多 40 epochs，验证 Macro-F1 patience 10
- 测试集只用于冻结 checkpoint 后的最终评价

## 结果

{table}

最佳消融模型为 {best['variant']}，测试 Macro-F1={best['macro_f1']:.6f}。
M3 相对 M0 的 Macro-F1 变化为 {full - baseline:+.6f}。

`ablation_macro_micro_f1.svg` 给出 Macro/Micro-F1 对比图。
"""
    (output_dir / "ablation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print(f"Ablation summary written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize frozen M0-M3 DeepDDI ablation results."
    )
    parser.add_argument("--checkpoints-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summarize(
        args.checkpoints_root.resolve(),
        args.results_root.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
