from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)


class MetricsTable:
    def __init__(self):
        self.columns = []
        self.rows = []

    def add_column(self, col_name):
        """Adds a column header to the table."""
        if col_name not in self.columns:
            self.columns.append(col_name)

    def add_rows(self, list_of_dicts):
        """
        Accepts a list of dictionaries and adds them to the table.
        Example: table.add_rows([{'epoch': 1, 'loss': 0.5}, {'epoch': 2, 'loss': 0.3}])
        """
        for row in list_of_dicts:
            # 1. Update columns dynamically if new keys appear
            for key in row.keys():
                if key not in self.columns:
                    self.columns.append(key)

            # 2. Add the row
            self.rows.append(row)

    def add_entry(self, **kwargs):
        """Adds a single row using keyword arguments."""
        self.add_rows([kwargs])

    def display(self):
        """Prints the table formatted nicely to the console."""
        if not self.columns:
            print("Table is empty.")
            return

        # 1. Calculate column widths
        col_widths = {col: len(col) for col in self.columns}
        for row in self.rows:
            for col in self.columns:
                val = str(row.get(col, ""))
                col_widths[col] = max(col_widths[col], len(val))

        # 2. Create format string
        fmt = "  ".join([f"{{:<{col_widths[col]}}}" for col in self.columns])

        # 3. Print Table
        separator = "-" * (sum(col_widths.values()) + 2 * (len(self.columns) - 1))

        print(separator)
        print(fmt.format(*self.columns))
        print(separator)

        for row in self.rows:
            row_values = [str(row.get(col, "-")) for col in self.columns]
            print(fmt.format(*row_values))
        print("")


def compute_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, name: str):
    # Computes and visualizes Confusion Matrix
    # Inspired by the A4_3 notebook

    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix for holdout
    labels = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=True, ax=ax, values_format="d")
    ax.set_title(f"{name} (Holdout)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()


def compute_reliability_diagramm(clf_dict: dict, save_dir=None):
    # Ispired from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(6, 2)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}

    for i, (name, (y_true, y_prob)) in enumerate(clf_dict.items()):
        display = CalibrationDisplay.from_predictions(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration Curves")

    # --- Histograms (1 per classifier, max 4 shown) ---
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1)]
    max_plots = min(len(clf_dict), 8)

    for idx, ((name, (_, y_prob)), (row, col)) in zip(
        range(max_plots), zip(clf_dict.items(), grid_positions)
    ):
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(idx),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()

    if save_dir is not None:
        timestamp = int(time.time())
        plt.savefig(Path(save_dir) / f"reliability_diagram_{timestamp}.png")

    plt.show()


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for multiclass classification.

    Args:
        y_true (np.array): Ground truth labels (integers), shape (n_samples,).
        y_prob (np.array): Predicted probabilities, shape (n_samples, n_classes).
        n_bins (int): Number of bins to use for calibration.

    Returns:
        float: The ECE score (lower is better).
    """
    if y_prob is None:
        return float("nan")
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2:
        return float("nan")

    # 1. Get the predicted class and the associated confidence (probability)
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)

    # 2. Check where predictions match ground truth (accuracy)
    accuracies = predictions == y_true

    # 3. Define bin boundaries (evenly spaced from 0 to 1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        # Define the lower and upper bound for this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Select samples whose confidence falls into this bin
        # Note: We include the lower bound for the first bin, otherwise > lower
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = np.sum(in_bin)

        if bin_count > 0:
            # Average accuracy in this bin
            avg_acc = np.mean(accuracies[in_bin])

            # Average confidence in this bin
            avg_conf = np.mean(confidences[in_bin])

            # Weighted absolute difference
            ece += (bin_count / n_samples) * np.abs(avg_acc - avg_conf)

    return ece


def average_iir_latency_ms(fs=250.0, fmin=8.0, fmax=30.0, n_points=100):
    """
    Compute average latency (ms) of an IIR SOS filter over its passband.

    Parameters:
    - sos: SOS matrix from scipy.signal.butter (or similar)
    - fs: sampling rate in Hz
    - fmin, fmax: passband edges in Hz
    - n_points: number of frequency points to average over

    Returns:
    - avg_latency_ms: average latency in milliseconds
    """
    from scipy.signal import group_delay, sos2tf

    lowcut = 8.0
    highcut = 30.0
    frequencies = [lowcut, highcut]
    order = 4
    fs = 250.0
    # Convert SOS to transfer function
    sos = scipy.signal.butter(
        order, Wn=np.array(frequencies), btype="bandpass", fs=fs, output="sos"
    )
    b, a = sos2tf(sos)

    # Compute group delay for each frequency
    w, gd = group_delay((b, a), fs=fs)
    freqs = np.linspace(fmin, fmax, n_points)

    # Interpolate group delay at desired passband frequencies
    gd_passband = np.interp(freqs, w, gd)

    # Average latency in milliseconds
    avg_latency_ms = np.mean(gd_passband) / fs * 1000

    return avg_latency_ms


def compute_itr(n_classes, accuracy, time_per_trial):
    """
    Computes Information Transfer Rate (ITR) based on Wolpaw's definition.

    Args:
        n_classes (int): Number of targets/classes (N).
        accuracy (float): Classification accuracy (P) in range [0, 1].
        time_per_trial (float): Time in seconds per selection/trial.

    Returns:
        dict: A dictionary containing:
            - 'itr_bits_per_min': ITR in bits/minute.
            - 'bits_per_trial': Information per trial in bits.
    """
    # 1. Check constraints
    if n_classes < 2:
        raise ValueError("Number of classes must be >= 2")
    if not (0 <= accuracy <= 1):
        raise ValueError("Accuracy must be between 0 and 1")

    # 2. If accuracy is less than random chance (1/N), ITR is 0.
    if accuracy < (1.0 / n_classes):
        return {"itr_bits_per_min": 0.0, "bits_per_trial": 0.0}

    # 3. Handle Perfect Accuracy Case (to avoid log(0) error)
    if accuracy == 1.0:
        bits_per_trial = math.log2(n_classes)

    else:
        # Wolpaw's formula
        term1 = math.log2(n_classes)
        term2 = accuracy * math.log2(accuracy)
        term3 = (1 - accuracy) * math.log2((1 - accuracy) / (n_classes - 1))

        bits_per_trial = term1 + term2 + term3

    # 4. Convert to Bits Per Minute
    # (60 seconds / time per trial) * bits per trial
    itr_per_min = bits_per_trial * (60.0 / time_per_trial)

    return {"itr_bits_per_min": itr_per_min, "bits_per_trial": bits_per_trial}


def compile_metrics(y_true, y_pred, y_prob, timings, n_classes):
    """Compute all metrics for a given set of predictions."""
    metrics = {}

    # Performance Metrics
    metrics["Acc."] = round(accuracy_score(y_true, y_pred), 4)
    metrics["F1 Score"] = round(f1_score(y_true, y_pred, average="macro"), 4)
    metrics["B. Acc."] = round(balanced_accuracy_score(y_true, y_pred), 4)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    metrics["ECE"] = round(ece, 4) if np.isfinite(ece) else float("nan")

    # Brier score:
    # - sklearn's brier_score_loss is binary-only.
    # - Here we compute a multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2
    # - Use direct label indexing so Brier works when validation fold has a subset of classes
    #   (e.g. leave-one-session-out CV where a held-out session may not contain all classes)
    brier = float("nan")
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[0] == len(y_true):
            k = y_prob_arr.shape[1]
            y_true = np.asarray(y_true, dtype=np.intp)
            if k >= 1 and np.all((y_true >= 0) & (y_true < k)):
                y_onehot = np.zeros((len(y_true), k), dtype=float)
                y_onehot[np.arange(len(y_true)), y_true] = 1.0
                brier = float(np.mean(np.sum((y_prob_arr - y_onehot) ** 2, axis=1)))
    metrics["Brier"] = round(brier, 4) if np.isfinite(brier) else float("nan")

    # Timings
    if timings is not None:
        metrics["Train Time (s)"] = round(timings["train_time"], 2)

        # Delay added by the filtering process
        metrics["Avg. Filter Latency (ms)"] = round(timings["filter_latency"], 2)

        # Time it takes to classifiy from raw to prediction (preprocessing + inference)
        metrics["Avg. Infer Latency (ms)"] = round(timings["infer_latency"], 2)

        # Time it takes to send a command to the game (preprocessing, classification, transfer) ??
        metrics["Avg. Total Latency (ms)"] = round(timings["total_latency"], 2)

        # TODO: Check again if this is correct
        itr = compute_itr(
            n_classes=n_classes,
            accuracy=metrics["Acc."],
            time_per_trial=(
                metrics["Avg. Infer Latency (ms)"] + timings["filter_latency"]
            )
            / 1000,  # TODO: Is this correct?
        )
        metrics["ITR (bits/min)"] = round(itr["itr_bits_per_min"], 2)

    return metrics


def print_online_metrics(timings):
    """Compute all metrics relevant during the online phase."""
    metrics = {}
    # Delay added by the filtering process
    metrics["Avg. Filter Latency (ms)"] = round(timings["filter_latency"], 2)

    # Time it takes to classifiy from raw to prediction (preprocessing + inference)
    metrics["Avg. Infer Latency (ms)"] = round(timings["infer_latency"], 2)

    # Time it takes to send a command to the game (preprocessing, classification, transfer) ??
    metrics["Avg. Total Latency (ms)"] = round(timings["total_latency"], 2)

    pass
