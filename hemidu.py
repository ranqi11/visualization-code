import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

# =================================================
# 0. 用户自定义参数（请根据需求修改）
# =================================================
# 数据集路径
label_root = "D:\BaiduNetdiskDownload\KITTI_split_3_6.5_3_0.5\labels"

# 图像分辨率（用于将归一化坐标转为像素）
IMG_W = 1242
IMG_H = 375

# ---- 坐标轴显示范围（手动指定，若设为 None 则自动计算）----
# xlim_manual = [0, 400]      # [xmin, xmax] 宽度范围
# ylim_manual = [0, 400]      # [ymin, ymax] 高度范围
xlim_manual = None
ylim_manual = None
# ---- 阈值线（单位：像素）----
small_max = 32   # 小目标上限
medium_max = 96  # 中目标上限（大于该值为大目标）

# ---- 局部放大图设置 ----
enable_inset = True                     # 是否显示局部放大图
inset_xlim = [0, 100]                   # 放大图的 X 轴范围
inset_ylim = [0, 100]                   # 放大图的 Y 轴范围
inset_loc = 'lower right'               # 放大图在主图中的位置
inset_width = 2                       # 放大图宽度（英寸）
inset_height = 2                      # 放大图高度（英寸）

# 其他绘图样式参数
scatter_alpha = 0.35
scatter_size = 8
contour_levels = 8
contour_color = "#E24A33"
grid_alpha = 0.5

# =================================================
# 1. 读取所有标签文件，计算宽度和高度（像素）
# =================================================
width_list = []
height_list = []

for sub in ["train", "val", "test"]:
    sub_dir = os.path.join(label_root, sub)
    if not os.path.exists(sub_dir):
        print(f"[警告] 未找到目录：{sub_dir}，跳过。")
        continue
    print(f"读取 {sub} 中的标签...")
    for txt in os.listdir(sub_dir):
        if not txt.endswith(".txt"):
            continue
        path = os.path.join(sub_dir, txt)
        with open(path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                width_px = w * IMG_W
                height_px = h * IMG_H
                width_list.append(width_px)
                height_list.append(height_px)

width = np.array(width_list)
height = np.array(height_list)

print("=====================================")
print(f"读取完成，共目标数量：{len(width)}")
print(f"宽度范围：{width.min():.1f} ~ {width.max():.1f}")
print(f"高度范围：{height.min():.1f} ~ {height.max():.1f}")

# 自动计算轴范围（如果未手动指定）
if xlim_manual is None:
    xlim = (0, width.max() * 1.02)
else:
    xlim = xlim_manual
if ylim_manual is None:
    ylim = (0, height.max() * 1.02)
else:
    ylim = ylim_manual

print(f"显示范围 X: {xlim}, Y: {ylim}")
print(f"阈值：小目标 ≤ {small_max}px, 中目标 {small_max+1}~{medium_max}px, 大目标 > {medium_max}px")
print("=====================================")

# =================================================
# 2. 创建图形和主坐标轴
# =================================================
fig = plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
gs = fig.add_gridspec(
    3, 3,
    width_ratios=[1, 4, 1],
    height_ratios=[1, 4, 1],
    left=0.08, right=0.92, bottom=0.08, top=0.92,
    wspace=0.12, hspace=0.12
)

ax_main = fig.add_subplot(gs[1, 1], facecolor="white")
ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main, facecolor="#FCFCFC")
ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main, facecolor="#FCFCFC")

# 设置主图显示范围
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)

# ----- 散点图 -----
ax_main.scatter(width, height, c="#9E9E9E", alpha=scatter_alpha, s=scatter_size, edgecolors="none")

# ----- KDE 密度估计（仅计算当前显示区域内的网格）-----
mask = (width >= xlim[0]) & (width <= xlim[1]) & (height >= ylim[0]) & (height <= ylim[1])
if mask.sum() < 10:
    print("警告：显示区域内数据点过少，跳过 KDE 绘制")
else:
    data_points = np.vstack([width[mask], height[mask]])
    try:
        kde = gaussian_kde(data_points)
        n_grid = 80
        X, Y = np.mgrid[xlim[0]:xlim[1]:n_grid*1j, ylim[0]:ylim[1]:n_grid*1j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)
        ax_main.contour(X, Y, Z, levels=contour_levels, colors=contour_color,
                        linewidths=1.4, alpha=0.85)
    except Exception as e:
        print(f"KDE 计算失败：{e}，跳过等高线")

# ----- 阈值线 -----
ax_main.axhline(small_max, color="#7DCEA0", linestyle="--", linewidth=1.5, alpha=0.9)
ax_main.axvline(small_max, color="#7DCEA0", linestyle="--", linewidth=1.5, alpha=0.9)
ax_main.axhline(medium_max, color="#5DADE2", linestyle="-.", linewidth=1.5, alpha=0.9)
ax_main.axvline(medium_max, color="#5DADE2", linestyle="-.", linewidth=1.5, alpha=0.9)

# ----- 网格 -----
ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=grid_alpha, color="#CCCCCC")

# ----- 标签 -----
ax_main.set_xlabel("Object Width (px)", fontsize=12, fontweight="bold", color="#333333")
ax_main.set_ylabel("Object Height (px)", fontsize=12, fontweight="bold", color="#333333")

# ----- 图例 -----
legend_elements = [
    Line2D([0], [0], color="#7DCEA0", linestyle="--", linewidth=1.5, label=f"Small (0 ~ {small_max}px)"),
    Line2D([0], [0], color="#5DADE2", linestyle="-.", linewidth=1.5, label=f"Medium ({small_max} ~ {medium_max}px)"),
    # Line2D([0], [0], color="#E24A33", linestyle="-", linewidth=1.5, label=f"Large (> {medium_max}px)"),
    Line2D([0], [0], color="#FFFFFF", linestyle="", linewidth=1.5, label=f"Large (> {medium_max}px)"),
]
ax_main.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

# =================================================
# 3. 边缘直方图（顶部和右侧）—— 完全不显示刻度
# =================================================
bins_x = np.linspace(xlim[0], xlim[1], 40)
ax_top.hist(width, bins=bins_x, range=xlim, color="#707F8E", edgecolor="white", alpha=0.7)
# 隐藏所有刻度线及标签
ax_top.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax_top.set_ylabel("")
for spine in ["top", "left", "right"]:
    ax_top.spines[spine].set_visible(False)
ax_top.spines["bottom"].set_color("#CCCCCC")

bins_y = np.linspace(ylim[0], ylim[1], 40)
ax_right.hist(height, bins=bins_y, range=ylim, orientation="horizontal",
              color="#707F8E", edgecolor="white", alpha=0.7)
# 隐藏所有刻度线及标签
ax_right.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax_right.set_xlabel("")
for spine in ["top", "bottom", "right"]:
    ax_right.spines[spine].set_visible(False)
ax_right.spines["left"].set_color("#CCCCCC")

# =================================================
# 4. 局部放大图（inset）
# =================================================
if enable_inset:
    ax_inset = inset_axes(ax_main, width=inset_width, height=inset_height,
                          loc=inset_loc, borderpad=1.5)
    ax_inset.set_xlim(inset_xlim)
    ax_inset.set_ylim(inset_ylim)
    ax_inset.set_facecolor("white")

    mask_inset = (width >= inset_xlim[0]) & (width <= inset_xlim[1]) & \
                 (height >= inset_ylim[0]) & (height <= inset_ylim[1])
    ax_inset.scatter(width[mask_inset], height[mask_inset],
                     c="#9E9E9E", alpha=scatter_alpha, s=scatter_size*1.5, edgecolors="none")

    if mask_inset.sum() >= 10:
        try:
            data_inset = np.vstack([width[mask_inset], height[mask_inset]])
            kde_inset = gaussian_kde(data_inset)
            n_grid_inset = 50
            Xi, Yi = np.mgrid[inset_xlim[0]:inset_xlim[1]:n_grid_inset*1j,
                              inset_ylim[0]:inset_ylim[1]:n_grid_inset*1j]
            pos_inset = np.vstack([Xi.ravel(), Yi.ravel()])
            Zi = np.reshape(kde_inset(pos_inset).T, Xi.shape)
            ax_inset.contour(Xi, Yi, Zi, levels=contour_levels, colors=contour_color,
                             linewidths=1.2, alpha=0.8)
        except:
            pass

    if inset_xlim[0] <= small_max <= inset_xlim[1]:
        ax_inset.axvline(small_max, color="#7DCEA0", linestyle="--", linewidth=1.2, alpha=0.9)
    if inset_ylim[0] <= small_max <= inset_ylim[1]:
        ax_inset.axhline(small_max, color="#7DCEA0", linestyle="--", linewidth=1.2, alpha=0.9)
    if inset_xlim[0] <= medium_max <= inset_xlim[1]:
        ax_inset.axvline(medium_max, color="#5DADE2", linestyle="-.", linewidth=1.2, alpha=0.9)
    if inset_ylim[0] <= medium_max <= inset_ylim[1]:
        ax_inset.axhline(medium_max, color="#5DADE2", linestyle="-.", linewidth=1.2, alpha=0.9)

    ax_inset.tick_params(labelsize=8)
    ax_inset.set_xlabel("Width (px)", fontsize=8)
    ax_inset.set_ylabel("Height (px)", fontsize=8)

    mark_inset(ax_main, ax_inset, loc1=1, loc2=3, fc="none", ec="#555555", lw=1.2, alpha=0.7)

# =================================================
# 5. 保存和显示
# =================================================
out_path = "yolo_bbox_distribution_custom.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.show()

print(f"图像已保存为：{out_path}")