#!/usr/bin/env python3
"""
Count class instances in YOLO format dataset and plot bar charts.
Configuration parameters are set in the main function.
"""

import os
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_class_names(file_path):
    """Load class names file, return list with index mapping to class_id."""
    if not file_path or not os.path.isfile(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names if names else None


def count_labels_in_folder(folder_path):
    """
    Count class occurrences in all YOLO label files within a folder.
    Returns Counter object: key=class_id, value=count.
    """
    counter = Counter()
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Warning: folder does not exist {folder_path}")
        return counter

    txt_files = list(folder_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        counter[class_id] += 1
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    return counter


def collect_statistics(labels_root, subsets):
    """Count class instances for each subset, return dict and sorted all class ids."""
    stats = {}
    all_classes = set()
    for subset in subsets:
        subset_path = os.path.join(labels_root, subset)
        cnt = count_labels_in_folder(subset_path)
        stats[subset] = cnt
        all_classes.update(cnt.keys())
    return stats, sorted(all_classes)


def plot_grouped_bar(stats, all_classes, class_names, output_path, dpi, figsize):
    """
    Grouped bar chart: each class has a group of bars (train/val/test).
    Uses Nature/Science style discrete colors.
    """
    subsets = list(stats.keys())
    n_classes = len(all_classes)
    n_subsets = len(subsets)

    bar_width = 0.8 / n_subsets
    x = np.arange(n_classes)

    # Nature/Science journal style colors - professional, muted, publication-ready
    # 参考顶刊常用配色：深蓝、砖红、橄榄绿、金色、蓝灰等
    colors = [
        "#1f77b4",  # 深蓝 - 经典科研蓝
        "#d62728",  # 砖红 - 醒目但不过于鲜艳
        "#2ca02c",  # 橄榄绿 - 自然协调
        "#ff7f0e",  # 橙色 - 温暖对比
        "#9467bd",  # 紫色 - 优雅
        "#8c564b",  # 棕褐 - 沉稳
        "#e377c2",  # 粉色 - 柔和点缀
        "#7f7f7f",  # 灰色 - 中性
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # 移除顶部和右侧边框，只保留坐标轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    bars_by_subset = []
    for i, subset in enumerate(subsets):
        counts = [stats[subset].get(cls, 0) for cls in all_classes]
        offset = (i - n_subsets / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            counts,
            width=bar_width,
            label=subset.capitalize(),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )
        bars_by_subset.append(bars)

    # Show values on top of bars
    for bars in bars_by_subset:
        ax.bar_label(bars, label_type="edge", fontsize=8, padding=2, fmt="%d")

    # X-axis labels
    if class_names and len(class_names) >= n_classes:
        xtick_labels = [class_names[cls] for cls in all_classes]
    else:
        xtick_labels = [f"Class {cls}" for cls in all_classes]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel("Number of instances", fontsize=11)
    ax.set_title("", fontsize=13, pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


def plot_aggregated_bar(stats, all_classes, class_names, output_path, dpi, figsize):
    """
    Aggregated bar chart: total instances per class, each bar with a distinct color.
    """
    total_counter = Counter()
    for cnt in stats.values():
        total_counter.update(cnt)

    classes = sorted(total_counter.keys())
    counts = [total_counter[cls] for cls in classes]
    n_bars = len(classes)

    # Nature/Science journal style colors - 均匀分布的不同色系
    # 蓝、红、绿、橙、紫、青、金、灰等，循环使用保证每个类别都有独特颜色
    base_colors = [
        "#1f77b4",  # 深蓝
        "#d62728",  # 砖红
        "#2ca02c",  # 橄榄绿
        "#ff7f0e",  # 橙色
        "#9467bd",  # 紫色
        "#17becf",  # 青色
        "#e377c2",  # 粉色
        "#8c564b",  # 棕褐
        "#7f7f7f",  # 灰色
        "#bcbd22",  # 黄绿
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(n_bars)]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # 移除顶部和右侧边框，只保留坐标轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    bars = ax.bar(
        range(len(classes)),
        counts,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )
    ax.bar_label(bars, label_type="edge", fontsize=9, padding=3, fmt="%d")

    if class_names and len(class_names) >= max(classes) + 1:
        xtick_labels = [class_names[cls] for cls in classes]
    else:
        xtick_labels = [f"Class {cls}" for cls in classes]

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel("Number of instances", fontsize=11)
    ax.set_title("", fontsize=13, pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


def main():
    # ==================== Configuration (edit directly) ====================
    labels_dir = r""  # root labels folder containing train/val/test
    output_path = "class3.png"             # output image path, e.g., "class_stats.png", None to display
    class_names_file = r""        # path to class names file (one per line), None to ignore
    aggregate = False             # True: total counts across all subsets; False: grouped by subset
    dpi = 600                      # output resolution
    figsize = (10, 6)              # figure size (width, height) in inches
    # ===================================================================

    subsets = ["train", "val", "test"]

    if not os.path.isdir(labels_dir):
        print(f"Error: labels root directory does not exist '{labels_dir}'")
        return

    stats, all_classes = collect_statistics(labels_dir, subsets)

    if not all_classes:
        print("No class data found. Check labels directory structure and .txt files.")
        return

    # Console output
    print("\n=== Class Statistics ===")
    for subset, cnt in stats.items():
        if cnt:
            print(f"\n{subset.upper()}:")
            for cls in sorted(cnt.keys()):
                print(f"  Class {cls}: {cnt[cls]} instances")
        else:
            print(f"\n{subset.upper()}: no data")

    class_names = load_class_names(class_names_file) if class_names_file else None

    if aggregate:
        plot_aggregated_bar(stats, all_classes, class_names, output_path, dpi, figsize)
    else:
        non_empty_subsets = [s for s in subsets if stats[s]]
        if not non_empty_subsets:
            print("No data in any subset, cannot plot.")
            return
        filtered_stats = {s: stats[s] for s in non_empty_subsets}
        plot_grouped_bar(filtered_stats, all_classes, class_names, output_path, dpi, figsize)


if __name__ == "__main__":
    main()