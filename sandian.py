import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# -------------------------------------------------
# 1. 设置 labels 根目录（内部必须有 train/val/test）
# -------------------------------------------------
label_root = ""   # 

# -------------------------------------------------
# 2. 设置统一分辨率
# -------------------------------------------------
IMG_W = 1242   # 图片宽度
IMG_H = 375    # 图片高度

# -------------------------------------------------
# 3. 自动读取所有标签文件
# -------------------------------------------------
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
                width_px  = w * IMG_W
                height_px = h * IMG_H
                width_list.append(width_px)
                height_list.append(height_px)

width = np.array(width_list)
height = np.array(height_list)

print("=====================================")
print("读取完成，共目标数量：", len(width))
print("宽度范围：", width.min(), "~", width.max())
print("高度范围：", height.min(), "~", height.max())
print("（注：后续绘图将只显示 0~400 区域）")
print("=====================================")

# -------------------------------------------------
# 4. 美化绘图（y轴0~400，网页背景，柔和配色）
# -------------------------------------------------
fig = plt.figure(figsize=(10, 10), facecolor="#F2F2F2")   # 网页浅灰底色

# 调整间距使边缘直方图与主图分隔更明显
gs = fig.add_gridspec(
    3, 3,
    width_ratios=[1, 4, 1],
    height_ratios=[1, 4, 1],
    left=0.08, right=0.92, bottom=0.08, top=0.92,
    wspace=0.12, hspace=0.12          # 增大分隔
)

ax_main = fig.add_subplot(gs[1, 1], facecolor="white")
ax_top  = fig.add_subplot(gs[0, 1], sharex=ax_main, facecolor="#FCFCFC")
ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main, facecolor="#FCFCFC")

# ----- 强制坐标轴范围 0~400 -----
ax_main.set_xlim(0, 400)
ax_main.set_ylim(0, 400)

# ----- 散点图（柔和灰色，高透明度）-----
ax_main.scatter(width, height, c="#9E9E9E", alpha=0.35, s=8, edgecolors="none")

# ----- KDE 密度估计（只计算 0~400 区域）-----
data_points = np.vstack([width, height])
kde = gaussian_kde(data_points)

X, Y = np.mgrid[0:400:100j, 0:400:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

# 使用柔和的玫瑰色等高线
ax_main.contour(X, Y, Z, levels=6, colors="#F1948A", linewidths=1.5, alpha=0.8)

# ----- 阈值线（柔和配色）-----
ax_main.axhline(16, color="#F5B041", linestyle=":",  linewidth=2, alpha=0.9)
ax_main.axvline(16, color="#F5B041", linestyle=":",  linewidth=2, alpha=0.9)
ax_main.axhline(32, color="#7DCEA0", linestyle="--", linewidth=1.5, alpha=0.9)
ax_main.axvline(32, color="#7DCEA0", linestyle="--", linewidth=1.5, alpha=0.9)
ax_main.axhline(96, color="#5DADE2", linestyle="-.", linewidth=1.5, alpha=0.9)
ax_main.axvline(96, color="#5DADE2", linestyle="-.", linewidth=1.5, alpha=0.9)

# ----- 网页风格网格（浅灰色，虚线）-----
ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#CCCCCC")

# ----- 坐标轴标签（加粗，深灰色）-----
ax_main.set_xlabel("Object Width (px)", fontsize=12, fontweight="bold", color="#333333")
ax_main.set_ylabel("Object Height (px)", fontsize=12, fontweight="bold", color="#333333")

# ----- 图例（柔和色调）-----
legend_elements = [
    Line2D([0], [0], color="#F5B041", linestyle=":",  linewidth=2, label="Micro < 16px"),
    Line2D([0], [0], color="#7DCEA0", linestyle="--", linewidth=1.5, label="Small < 32px"),
    Line2D([0], [0], color="#5DADE2", linestyle="-.", linewidth=1.5, label="Medium < 96px"),
]
ax_main.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

# ----- 顶部直方图（宽度，只显示 0~400）-----
bins = np.linspace(0, 400, 40)
ax_top.hist(width, bins=bins, range=(0, 400), color="#B0C4DE", edgecolor="white", alpha=0.7)
ax_top.set_yticks([])
ax_top.set_ylabel("")
# 隐藏顶部子图的上、左、右 spines，使其更简洁
for spine in ["top", "left", "right"]:
    ax_top.spines[spine].set_visible(False)

# ----- 右侧直方图（高度，只显示 0~400）-----
ax_right.hist(height, bins=bins, range=(0, 400), orientation="horizontal",
              color="#B0C4DE", edgecolor="white", alpha=0.7)
ax_right.set_xticks([])
ax_right.set_xlabel("")
# 隐藏右侧子图的上、下、右 spines
for spine in ["top", "bottom", "right"]:
    ax_right.spines[spine].set_visible(False)

# 可选：为直方图添加浅色边框线，使其与主图视觉分隔更明显
ax_top.spines["bottom"].set_color("#CCCCCC")
ax_right.spines["left"].set_color("#CCCCCC")

# ----- 保存图像（高DPI）-----
out_path = "yolo_bbox_distribution.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("图像已保存为：", out_path)