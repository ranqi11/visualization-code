"""
Robustness Analysis Boxplot for Academic Paper (Single Metric)
Author: Claude
Modified: 2026-04-08

Features:
- Top-tier journal color schemes
- Boxplot with overlaid scatter points
- Professional typography
- Export to high-quality PDF and PNG (both saved simultaneously)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# ============================================================
# Configuration: Top-tier Journal Style Settings
# ============================================================

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# ============================================================
# Color Scheme (Nature Style)
# ============================================================
COLORS_NATURE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']

# Option 2: Science Style (Modern & Vibrant)
COLORS_SCIENCE = ['#0C5DA5', '#FF6B6B', '#00B945', '#FF9500', '#845EC2']

# Option 3: IEEE Style (Professional & Clean)
COLORS_IEEE = ['#0072BD', '#D95319', '#EDB120', '#77AC30', '#4DBEEE']

# Option 4: Soft Academic Style (Gentle & Readable)
COLORS_SOFT = ['#6A9BD1', '#E88A8A', '#7BC89C', '#D4A574', '#9B8ACB']
COLORS = COLORS_SCIENCE

# ============================================================
# Data: mAP results across 5 random seeds for different methods
# ============================================================


data_mAP= {}
# ============================================================
# Utility Functions
# ============================================================

def calculate_statistics(data_dict):
    """Calculate mean and std for each method."""
    stats = {}
    for method, values in data_dict.items():
        stats[method] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'cv': np.std(values, ddof=1) / np.mean(values) * 100  # Coefficient of variation
        }
    return stats


def print_statistics(data_dict, metric_name):
    """Print statistics in a formatted table."""
    stats = calculate_statistics(data_dict)

    print(f"\n{'='*60}")
    print(f"Statistics for {metric_name}")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'CV(%)':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*60}")

    for method, s in stats.items():
        print(f"{method:<20} {s['mean']:>8.2f} {s['std']:>8.2f} "
              f"{s['cv']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")
    print(f"{'='*60}\n")

    return stats


def plot_boxplot(data_dict, ylabel='mAP (%)', title='Robustness Analysis of Detection Performance',
                 figsize=(10, 6), colors=None, show_scatter=True,
                 save_path=None, dpi=600):
    """
    Plot professional boxplot with scatter overlay.
    Saves both PDF and PNG if save_path is provided.

    Parameters
    ----------
    save_path : str, optional
        Base file path (without extension or with .pdf). Will generate .pdf and .png.
    dpi : int
        Resolution for PNG and PDF (PDF respects vector quality, dpi mainly for PNG).
    """
    if colors is None:
        colors = COLORS

    methods = list(data_dict.keys())
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=figsize)

    data_values = [data_dict[method] for method in methods]

    # Create boxplot
    bp = ax.boxplot(data_values,
                    tick_labels=methods,
                    patch_artist=True,
                    widths=0.5,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='gray', linewidth=1.2),
                    capprops=dict(color='gray', linewidth=1.2))

    # Color boxes
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors[:n_methods])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    # Overlay scatter points with jitter
    if show_scatter:
        for i, method in enumerate(methods):
            x = np.random.normal(i + 1, 0.08, size=len(data_dict[method]))
            ax.scatter(x, data_dict[method],
                      c=colors[i % len(colors)],
                      s=40, alpha=0.8,
                      edgecolors='black', linewidths=0.5,
                      zorder=3)

    # Add mean markers
    means = [np.mean(data_dict[method]) for method in methods]
    ax.scatter(range(1, n_methods + 1), means,
               marker='D', s=60, c='white', edgecolors='black',
               linewidths=1.5, zorder=4, label='Mean')

    # Add overall mean line
    overall_mean = np.mean([v for values in data_dict.values() for v in values])
    ax.axhline(y=overall_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel('', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    if n_methods > 4:
        ax.set_xticklabels(methods, rotation=15, ha='right')
    else:
        ax.set_xticklabels(methods)

    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()

    # Save both PDF and PNG if a save_path is given
    if save_path:
        # Determine base filename (strip extension if any)
        base = os.path.splitext(save_path)[0]
        pdf_path = base + '.pdf'
        png_path = base + '.png'

        plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figures saved to: {pdf_path} and {png_path} (DPI={dpi})")

    return fig, ax


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS - BOXPLOT VISUALIZATION (mAP Only)")
    print("="*70)

    # 1. Print statistics
    stats_mAP = print_statistics(data_mAP, "mAP")

    # 2. Plot mAP boxplot (saves both PDF and PNG with DPI=300)
    fig, ax = plot_boxplot(
        data_mAP,
        ylabel='mAP (%)',
        title='Robustness Analysis of Detection Performance (mAP)',
        save_path='boxplot_mAP.pdf',
        dpi=300          # DPI for PNG (PDF is vector, but dpi used for rasterized elements)
    )

    # 3. Summary for paper
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)

    best_method = max(stats_mAP.items(), key=lambda x: x[1]['mean'])
    print(f"Best performing method: {best_method[0]} with mAP = {best_method[1]['mean']:.2f} ± {best_method[1]['std']:.2f}%")
    print(f"Lowest CV (most stable): {min(stats_mAP.items(), key=lambda x: x[1]['cv'])[0]}")

    plt.show()
    print("\nDone! PDF and PNG saved with DPI=600.")