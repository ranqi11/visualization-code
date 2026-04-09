import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
# Configuration
# ============================================================================

# 5 metrics for radar chart
metrics = ['Precision', 'Recall', 'Map@50', 'Map@50-95', 'Params']

# 9 datasets to compare (example data, replace with your actual values)
data = {
 
}

# ============================================================================
# Academic Color Schemes (Top-tier Conference Style)
# ============================================================================

# Nature style (Vibrant & Professional)
COLORS_NATURE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                 '#8491B4', '#91D1C2', '#DC0000', '#7E6148']

# Science style (Modern & Vibrant)
COLORS_SCIENCE = ['#0C5DA5', '#FF6B6B', '#00B945', '#FF9500', '#845EC2',
                  '#7FC4FC', '#D4AF37', '#9B4DCA', '#4ECDC4']

# IEEE style (Professional & Clean)
COLORS_IEEE = ['#0072BD', '#D95319', '#EDB120', '#77AC30', '#4DBEEE',
               '#A2142F', '#7B2F8B', '#FF69B4', '#8B4513']

# Soft Academic style (Gentle & Readable)
COLORS_SOFT = ['#6A9BD1', '#E88A8A', '#7BC89C', '#D4A574', '#9B8ACB',
               '#B8D4E8', '#F2B8B8', '#A8D5BA', '#C9B8D4']

# NeurIPS/ACL style (Classic academic)
COLORS_NEURIPS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Choose a color scheme
COLORS = COLORS_NATURE

# ============================================================================
# Data Normalization
# ============================================================================
def normalize_data(data_dict, metrics):
    """Normalize data to [0, 1] scale for radar chart."""
    normalized = {}

    values_by_metric = {m: [] for m in metrics}
    for name, values in data_dict.items():
        for i, m in enumerate(metrics):
            values_by_metric[m].append(values[i])

    mins = {m: min(values_by_metric[m]) for m in metrics}
    maxs = {m: max(values_by_metric[m]) for m in metrics}

    for name, values in data_dict.items():
        normalized[name] = []
        for i, m in enumerate(metrics):
            if maxs[m] - mins[m] > 0:
                if m != 'Params':
                    norm = (values[i] - mins[m]) / (maxs[m] - mins[m])
                else:
                    norm = 1 - (values[i] - mins[m]) / (maxs[m] - mins[m])
            else:
                norm = 0.5
            normalized[name].append(norm)

    return normalized, mins, maxs

# ============================================================================
# Radar Chart Function
# ============================================================================
def create_radar_chart(data_dict, metrics, colors, output_path='radar_chart.png',
                       dpi=300, show_values_for=None, color_scheme='NATURE'):
    """
    Create a professional radar chart for academic papers.

    Parameters:
    - data_dict: Dictionary with method names as keys and metric values as lists
    - metrics: List of metric names
    - colors: List of color codes
    - output_path: Output file path
    - dpi: Resolution for PNG output
    - show_values_for: If specified, show actual values for this method on the chart
    - color_scheme: Color scheme name for reference
    """
    # Select color scheme
    color_schemes = {
        'NATURE': COLORS_NATURE,
        'SCIENCE': COLORS_SCIENCE,
        'IEEE': COLORS_IEEE,
        'SOFT': COLORS_SOFT,
        'NEURIPS': COLORS_NEURIPS,
    }
    colors = color_schemes.get(color_scheme, COLORS_NATURE)

    # Normalize data
    normalized, mins, maxs = normalize_data(data_dict, metrics)

    # Set up the figure
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
    })
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Draw background circles (grid)
    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [level] * (num_vars + 1), 'k-', linewidth=0.5, alpha=0.3)

    # Draw outer boundary circle (thicker)
    ax.plot(angles, [1.0] * (num_vars + 1), 'k-', linewidth=1.2, alpha=0.5)

    # Draw axis lines
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], 'k-', linewidth=0.5, alpha=0.3)

    # Set axis labels - moved outside the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12, fontweight='medium')
    ax.xaxis.set_tick_params(pad=16)  # Move labels outward

    # Remove radial labels
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    # Plot each dataset
    for idx, (name, values) in enumerate(normalized.items()):
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2,
                label=name, color=colors[idx % len(colors)], markersize=5,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.fill(angles, values_plot, alpha=0.08, color=colors[idx % len(colors)])

    # Show specific values on chart if requested
    if show_values_for and show_values_for in data_dict:
        values = data_dict[show_values_for]
        normalized_values = normalized[show_values_for]
        values_plot = normalized_values + normalized_values[:1]

        # Add value labels at each vertex - closer to center
        for i, (angle, val, raw_val) in enumerate(zip(angles[:-1], normalized_values, values)):
            if metrics[i] == 'Params':
                label = f'{int(raw_val):,}'
            else:
                label = f'{raw_val:.3f}'
            # Position the label closer to center
            ax.annotate(label, xy=(angle, val), xytext=(angle, val - 0.08),
                       ha='center', va='top', fontsize=10,
                       color='black', fontweight='medium',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='none', alpha=0.7))

    # Add legend below the chart
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       ncol=min(5, len(data_dict)), frameon=True,
                       fancybox=False, edgecolor='#333333', fontsize=9)

    # Customize appearance
    ax.spines['polar'].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

    # Title
    plt.title('', size=13, fontweight='bold', pad=15)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to {output_path} (dpi={dpi})")
    print(f"Using color scheme: {color_scheme}")

    return fig

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Example: show values for '0.75:0.25' method
    create_radar_chart(
        data, metrics, COLORS,
        dpi=600,
        show_values_for='ours',  # 显示指定方法的数值
        color_scheme='SCIENCE'  # 可选: NATURE, SCIENCE, IEEE, SOFT, NEURIPS
    )