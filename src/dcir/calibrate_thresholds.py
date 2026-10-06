from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .evaluate_cloud import compute_metrics


def load_predictions(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    path = directory / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        target = np.asarray(archive["target"])
        probability = np.asarray(archive["probability"], dtype=np.float64)
    if target.shape != probability.shape:
        raise ValueError(
            f"Multilabel target/probability shape mismatch in {path}: "
            f"{target.shape} versus {probability.shape}"
        )
    if target.ndim != 2:
        raise ValueError("Threshold calibration requires 2-D multilabel arrays")
    if not np.isfinite(probability).all():
        raise ValueError(f"Non-finite probabilities found in {path}")
    return target.astype(np.int8), probability


def threshold_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    if not 0.0 < minimum < maximum < 1.0:
        raise ValueError("Require 0 < minimum < maximum < 1")
    if not 0.0 < step <= maximum - minimum:
        raise ValueError("step must be positive and within the search interval")
    count = int(np.floor((maximum - minimum) / step + 1e-9))
    values = minimum + np.arange(count + 1, dtype=np.float64) * step
    values = np.append(values, [0.5, maximum])
    return np.unique(np.round(values[(values >= minimum) & (values <= maximum)], 10))


def macro_f1_at_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float | np.ndarray,
) -> float:
    prediction = probability >= np.asarray(threshold)
    return float(
        f1_score(target, prediction, average="macro", zero_division=0)
    )


def _best_index(scores: np.ndarray, values: np.ndarray, anchor: float) -> int:
    best = float(scores.max())
    candidates = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
    return int(candidates[np.argmin(np.abs(values[candidates] - anchor))])


def fit_thresholds(
    target: np.ndarray,
    probability: np.ndarray,
    grid: np.ndarray,
    shrinkage: float = 20.0,
    min_support: int = 5,
) -> tuple[float, np.ndarray, list[dict]]:
    if target.shape != probability.shape:
        raise ValueError("target and probability shapes must match")
    if shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative")
    if min_support < 1:
        raise ValueError("min_support must be positive")

    global_scores = np.asarray(
        [
            macro_f1_at_threshold(target, probability, threshold)
            for threshold in grid
        ]
    )
    global_threshold = float(
        grid[_best_index(global_scores, grid, anchor=0.5)]
    )

    calibrated = np.full(target.shape[1], global_threshold, dtype=np.float64)
    rows: list[dict] = []
    for class_index in range(target.shape[1]):
        truth = target[:, class_index]
        score = probability[:, class_index]
        support = int(truth.sum())
        class_scores = np.asarray(
            [
                f1_score(
                    truth,
                    score >= threshold,
                    zero_division=0,
                )
                for threshold in grid
            ],
            dtype=np.float64,
        )
        raw_threshold = float(
            grid[
                _best_index(
                    class_scores,
                    grid,
                    anchor=global_threshold,
                )
            ]
        )
        if support < min_support:
            weight = 0.0
        elif shrinkage == 0.0:
            weight = 1.0
        else:
            weight = support / (support + shrinkage)
        final_threshold = (
            weight * raw_threshold
            + (1.0 - weight) * global_threshold
        )
        calibrated[class_index] = final_threshold
        rows.append(
            {
                "class_internal": class_index,
                "validation_support": support,
                "global_threshold": global_threshold,
                "raw_best_threshold": raw_threshold,
                "shrinkage_weight": weight,
                "calibrated_threshold": float(final_threshold),
                "fixed_0.5_validation_f1": float(
                    f1_score(truth, score >= 0.5, zero_division=0)
                ),
                "raw_best_validation_f1": float(class_scores.max()),
                "calibrated_validation_f1": float(
                    f1_score(
                        truth,
                        score >= final_threshold,
                        zero_division=0,
                    )
                ),
            }
        )
    return global_threshold, calibrated, rows


def load_label_values(directory: Path, num_classes: int) -> list[str]:
    path = directory / "per_class_metrics.csv"
    if not path.is_file():
        return [str(index) for index in range(num_classes)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != num_classes:
        raise ValueError(
            f"{path} contains {len(rows)} classes, expected {num_classes}"
        )
    return [str(row.get("class_raw", index)) for index, row in enumerate(rows)]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(
    validation_dir: Path,
    test_dir: Path,
    output_dir: Path,
    minimum: float,
    maximum: float,
    step: float,
    shrinkage: float,
    min_support: int,
) -> None:
    validation_target, validation_probability = load_predictions(validation_dir)
    grid = threshold_grid(minimum, maximum, step)
    global_threshold, thresholds, threshold_rows = fit_thresholds(
        validation_target,
        validation_probability,
        grid,
        shrinkage=shrinkage,
        min_support=min_support,
    )
    # Test labels are loaded only after every threshold has been frozen from
    # validation data. They are used solely to report final metrics.
    test_target, test_probability = load_predictions(test_dir)
    if validation_probability.shape[1] != test_probability.shape[1]:
        raise ValueError("Validation and test class counts do not match")
    label_values = load_label_values(
        validation_dir, validation_probability.shape[1]
    )
    for row, label in zip(threshold_rows, label_values):
        row["class_raw"] = label

    validation_fixed, _, _ = compute_metrics(
        validation_target,
        validation_probability,
        "multilabel",
        0.5,
        label_values,
    )
    validation_global, _, _ = compute_metrics(
        validation_target,
        validation_probability,
        "multilabel",
        global_threshold,
        label_values,
    )
    validation_calibrated, validation_per_class, validation_prediction = (
        compute_metrics(
            validation_target,
            validation_probability,
            "multilabel",
            thresholds,
            label_values,
        )
    )
    test_fixed, _, _ = compute_metrics(
        test_target,
        test_probability,
        "multilabel",
        0.5,
        label_values,
    )
    test_calibrated, test_per_class, test_prediction = compute_metrics(
        test_target,
        test_probability,
        "multilabel",
        thresholds,
        label_values,
    )

    threshold_summary = {
        "selection_split": "validation",
        "application_split": "test",
        "grid": {
            "minimum": minimum,
            "maximum": maximum,
            "step": step,
            "count": int(len(grid)),
        },
        "shrinkage": shrinkage,
        "min_support": min_support,
        "global_threshold": global_threshold,
        "per_class_threshold_min": float(thresholds.min()),
        "per_class_threshold_median": float(np.median(thresholds)),
        "per_class_threshold_max": float(thresholds.max()),
        "validation_fixed_0.5": validation_fixed,
        "validation_global": validation_global,
        "validation_calibrated": validation_calibrated,
        "test_fixed_0.5": test_fixed,
        "test_calibrated": test_calibrated,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            threshold_summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            test_calibrated,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    write_rows(output_dir / "thresholds.csv", threshold_rows)
    write_rows(
        output_dir / "validation_per_class_metrics.csv",
        validation_per_class,
    )
    write_rows(output_dir / "per_class_metrics.csv", test_per_class)
    np.save(output_dir / "thresholds.npy", thresholds)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        target=validation_target,
        probability=validation_probability,
        prediction=validation_prediction,
        threshold=thresholds,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=test_target,
        probability=test_probability,
        prediction=test_prediction,
        threshold=thresholds,
    )

    print(
        json.dumps(
            {
                "global_threshold": global_threshold,
                "threshold_min": float(thresholds.min()),
                "threshold_median": float(np.median(thresholds)),
                "threshold_max": float(thresholds.max()),
                "validation_macro_f1_fixed_0.5": validation_fixed["macro_f1"],
                "validation_macro_f1_calibrated": validation_calibrated[
                    "macro_f1"
                ],
                "test_macro_f1_fixed_0.5": test_fixed["macro_f1"],
                "test_macro_f1_calibrated": test_calibrated["macro_f1"],
                "test_micro_f1_calibrated": test_calibrated["micro_f1"],
                "test_subset_accuracy_calibrated": test_calibrated[
                    "subset_accuracy"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Calibrated results written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit support-shrunk per-class thresholds on validation predictions "
            "and apply the frozen thresholds to test predictions."
        )
    )
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum", type=float, default=0.05)
    parser.add_argument("--maximum", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()
    calibrate(
        args.validation_dir.resolve(),
        args.test_dir.resolve(),
        args.output_dir.resolve(),
        args.minimum,
        args.maximum,
        args.step,
        args.shrinkage,
        args.min_support,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
