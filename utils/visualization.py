"""
visualization.py
----------------
Reusable plotting helpers for all three pipeline notebooks.
Each function returns a Matplotlib Figure so callers can call plt.show()
or save the figure as needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
PALETTE = "tab20"
FIG_DPI = 120


# ---------------------------------------------------------------------------
# 1. Label / class distribution
# ---------------------------------------------------------------------------

def plot_label_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Horizontal bar chart of class frequencies."""
    counts = y.value_counts().sort_values(ascending=True)
    colors = sns.color_palette(PALETTE, n_colors=len(counts))

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    bars = ax.barh(counts.index, counts.values, color=colors,
                   edgecolor="white", linewidth=0.5)

    # Annotate bars with count
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + counts.values.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Missing / null value summary
# ---------------------------------------------------------------------------

def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing Values per Column",
    figsize: tuple = (14, 5),
    max_cols: int = 40,
) -> plt.Figure:
    """Bar chart of null count per column (only columns that have nulls)."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(
        ascending=False).head(max_cols)

    if null_counts.empty:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=FIG_DPI)
        ax.text(0.5, 0.5, "✓ No missing values found", ha="center", va="center",
                fontsize=14, color="green", transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    colors = sns.color_palette("Reds_r", n_colors=len(null_counts))
    ax.bar(null_counts.index, null_counts.values,
           color=colors, edgecolor="white")
    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Null Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Feature importance bar chart (from Random Forest)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances_df: pd.DataFrame,
    n_features: int = 20,
    title: str = "Top Feature Importances (Random Forest – Gini)",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Horizontal bar chart of the top-N feature importances."""
    top = importances_df.head(n_features).copy()
    # ascending for horizontal bars
    top = top.sort_values("importance", ascending=True)

    colors = sns.color_palette("Blues_d", n_colors=len(top))

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    bars = ax.barh(top["feature"], top["importance"],
                   color=colors, edgecolor="white")

    for bar, val in zip(bars, top["importance"]):
        ax.text(
            bar.get_width() + top["importance"].max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_all_feature_importance(
    importances_df: pd.DataFrame,
    n_top: int = 20,
    title: str = "All Feature Importances (top highlighted)",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Bar chart of ALL features, with the top-N highlighted in a different color."""
    df = importances_df.sort_values(
        "importance", ascending=False).reset_index(drop=True)
    colors = ["#2196F3" if i < n_top else "#BDBDBD" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    ax.bar(df["feature"], df["importance"], color=colors,
           edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label=f"Top {n_top} selected"),
        Patch(facecolor="#BDBDBD", label="Not selected"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (14, 11),
    annot_threshold: int = 15,
) -> plt.Figure:
    """Seaborn heatmap of feature correlations.

    Annotations are only shown when the number of features ≤ *annot_threshold*
    to keep the chart readable.
    """
    corr = df.corr(numeric_only=True)
    annot = len(corr) <= annot_threshold

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.4 if annot else 0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Feature distributions (before vs after scaling)
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    X_raw: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_names: list,
    n_features: int = 6,
    figsize: tuple = (16, 8),
    title: str = "Feature Distributions Before vs After Scaling",
) -> plt.Figure:
    """Side-by-side KDE plots for a sample of features before and after
    MinMax scaling."""
    n = min(n_features, len(feature_names))
    fig, axes = plt.subplots(2, n, figsize=figsize, dpi=FIG_DPI)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    for i, feat in enumerate(feature_names[:n]):
        # Before
        axes[0, i].hist(X_raw[feat].dropna(), bins=50,
                        color="#607D8B", edgecolor="white", linewidth=0.3)
        axes[0, i].set_title(feat, fontsize=8, fontweight="bold")
        if i == 0:
            axes[0, i].set_ylabel("Before", fontsize=9)

        # After
        axes[1, i].hist(X_scaled_df[feat], bins=50,
                        color="#1976D2", edgecolor="white", linewidth=0.3)
        if i == 0:
            axes[1, i].set_ylabel("After (0–1)", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Training history (loss & accuracy)
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict,
    title: str = "Training History",
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """Line plot of training and validation loss / accuracy over epochs.

    Parameters
    ----------
    history : dict
        Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        Each value is a list of per-epoch values.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=FIG_DPI)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-",
             color="#E53935", label="Train Loss", lw=1.5, ms=4)
    ax1.plot(epochs, history["val_loss"],   "s--",
             color="#FB8C00", label="Val Loss",   lw=1.5, ms=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-",
             color="#1E88E5", label="Train Acc", lw=1.5, ms=4)
    ax2.plot(epochs, history["val_acc"],   "s--",
             color="#43A047", label="Val Acc",   lw=1.5, ms=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    normalize: bool = True,
) -> plt.Figure:
    """Annotated seaborn heatmap of a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (raw counts) from sklearn.
    class_names : list
        Ordered list of class label strings.
    normalize : bool
        If True, normalize rows to show recall per class (values 0–1).
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2f"
        cmap = "Blues"
    else:
        cm_plot = cm
        fmt = "d"
        cmap = "Blues"

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Evaluation metrics bar chart (accuracy, precision, recall, F1)
# ---------------------------------------------------------------------------

def plot_evaluation_metrics(
    metrics: dict,
    title: str = "Model Evaluation Metrics",
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """Grouped bar chart of per-class and overall evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Nested dict: ``{class_name: {precision, recall, f1-score}} ``
        plus an ``'accuracy'`` key (scalar float).
        This matches the output of
        ``sklearn.metrics.classification_report(output_dict=True)``.
    """
    # Extract per-class metrics (skip aggregated rows and the CM array)
    skip_keys = {"accuracy", "macro avg", "weighted avg", "confusion_matrix"}
    class_names, precisions, recalls, f1s = [], [], [], []

    for cls, vals in metrics.items():
        if cls in skip_keys:
            continue
        class_names.append(cls)
        precisions.append(vals["precision"])
        recalls.append(vals["recall"])
        f1s.append(vals["f1-score"])

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    bars_p = ax.bar(x - width, precisions, width,
                    label="Precision", color="#1976D2")
    bars_r = ax.bar(x,         recalls,   width,
                    label="Recall",    color="#43A047")
    bars_f = ax.bar(x + width, f1s,       width,
                    label="F1-Score",  color="#FB8C00")

    def _annotate(bars):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.005,
                f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    _annotate(bars_p)
    _annotate(bars_r)
    _annotate(bars_f)

    # Overall accuracy line
    if "accuracy" in metrics:
        acc = metrics["accuracy"]
        ax.axhline(acc, color="#E53935", linestyle="--", lw=1.5,
                   label=f"Overall Accuracy = {acc:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    return fig


def plot_metrics_radar(
    metrics: dict,
    title: str = "Per-Class Metrics Radar Chart",
    figsize: tuple = (9, 9),
) -> plt.Figure:
    """Spider / radar chart comparing precision, recall, F1 per class.

    Parameters
    ----------
    metrics : dict
        Same format as ``plot_evaluation_metrics``.
    """
    skip_keys = {"accuracy", "macro avg", "weighted avg", "confusion_matrix"}
    class_names, precisions, recalls, f1s = [], [], [], []

    for cls, vals in metrics.items():
        if cls in skip_keys:
            continue
        class_names.append(cls)
        precisions.append(vals["precision"])
        recalls.append(vals["recall"])
        f1s.append(vals["f1-score"])

    categories = ["Precision", "Recall", "F1-Score"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(polar=True), dpi=FIG_DPI)
    colors = sns.color_palette(PALETTE, n_colors=len(class_names))

    for i, cls in enumerate(class_names):
        values = [precisions[i], recalls[i], f1s[i]]
        values += values[:1]
        ax.plot(angles, values, "o-", lw=1.5, label=cls, color=colors[i])
        ax.fill(angles, values, alpha=0.07, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    return fig


def plot_roc_curves(
    y_true_onehot: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
    title: str = "ROC Curves (One-vs-Rest)",
    figsize: tuple = (9, 7),
) -> plt.Figure:
    """Plot one ROC curve per class (one-vs-rest strategy).

    Parameters
    ----------
    y_true_onehot : np.ndarray  shape (n, n_classes)
    y_prob : np.ndarray         shape (n, n_classes) – softmax probabilities
    class_names : list
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    colors = sns.color_palette(PALETTE, n_colors=len(class_names))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, color=color,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    return fig
