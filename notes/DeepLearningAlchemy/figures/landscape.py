"""Generate the original chapter 06–07 figures (requires NumPy and Matplotlib).

Run: python3 notes/DeepLearningAlchemy/figures/landscape.py
Set CS312_PLOT_FONT to a CJK font file if the default fonts are unavailable.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors, ticker
import numpy as np

assets = Path(__file__).resolve().parent.parent / "assets"
assets.mkdir(exist_ok=True)
font_path = Path(os.getenv("CS312_PLOT_FONT", "/System/Library/Fonts/STHeiti Medium.ttc"))
if font_path.is_file():
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
else:
    font_name = font_manager.FontProperties(
        family=["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei"]
    )
    font_name = font_manager.FontProperties(
        fname=font_manager.findfont(font_name, fallback_to_default=False)
    ).get_name()
ink = "#172c42"
blue = "#285faf"
teal = "#0f7163"
orange = "#b64920"
dim = "#4f6375"
paper = "#fbf9f4"

plt.rcParams.update({
    "font.family": [font_name, "DejaVu Sans"],
    "font.size": 16,
    "mathtext.fontset": "dejavusans",
    "axes.labelcolor": ink,
    "text.color": ink,
    "axes.edgecolor": "#aab5be",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.color": dim,
    "ytick.color": dim,
    "legend.fontsize": 15,
    "legend.frameon": False,
    "figure.facecolor": paper,
    "axes.facecolor": paper,
})

def save(fig, name):
    fig.savefig(assets / f"{name}.png", dpi=164, bbox_inches="tight",
                pad_inches=0.18, facecolor=paper)
    plt.close(fig)

def grid(ax):
    ax.grid(axis="y", color="#9bafbd", linewidth=1, alpha=0.26)
    ax.set_axisbelow(True)

def contour_plot(xlim, ylim, height=4.7):
    fig, ax = plt.subplots(figsize=(7.6, height), layout="constrained")
    x, y = np.meshgrid(np.linspace(*xlim, 240), np.linspace(*ylim, 240))
    ax.set(xlim=xlim, ylim=ylim, xlabel="参数坐标 $x$", ylabel="参数坐标 $y$")
    return fig, ax, x, y

def mark(ax, x, y, color, marker="o", size=90):
    ax.scatter(x, y, s=size, facecolor=color, marker=marker,
               edgecolor=paper, linewidth=1.5, zorder=9)

# Ch06. Exact level sets with a linear barrier and a zero-loss curved path.
fig, ax, x, y = contour_plot((-1.49, 1.49), (-0.72, 1.28))
loss = 0.5 * (x*x + y*y - 1)**2
ax.contourf(x, y, loss, levels=[0, .012, .06, .18, .38, .58, 1.1, 5],
            cmap=colors.LinearSegmentedColormap.from_list("loss",
                ["#e0f4e8", "#f3edda", "#e8ad87", "#b75a50"]))
ax.contour(x, y, loss, levels=[.02, .08, .2, .4, .6, 1.1],
           colors="#697478", alpha=.33, linewidths=.9)
angle = np.linspace(0, np.pi, 120)
ax.plot(np.cos(angle), np.sin(angle), color=teal, linewidth=3.8,
        label="弯曲路径")
ax.plot([-1, 1], [0, 0], "--", color=orange, linewidth=3.8,
        label="线性插值")
mark(ax, [-1, 1], [0, 0], teal)
mark(ax, 0, 0, orange, "D")
ax.text(-1, -.17, "A", ha="center")
ax.text(1, -.17, "B", ha="center")
ax.text(0, .13, "中点损失 0.50", ha="center", color=ink)
ax.text(0, -.46, "两端与整条绿色曲线的损失均为 0", ha="center", color=ink)
ax.set_title("低损失端点之间可能有直线障碍")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncols=2)
save(fig, "ch06-mode-connectivity")

# Ch06. SGD toy model with noise only across the valley.
rng = np.random.default_rng(32)
end, cooling = 120, 65
steps = np.arange(end + 1)
etas = np.full(end, 1.88)
etas[cooling:] = np.linspace(1.88, .08, end - cooling)
valley_x, valley_y = np.zeros(end + 1), np.zeros(end + 1)
valley_y[0] = .55
for i, (eta, noise) in enumerate(zip(etas, rng.normal(0, .10, end))):
    valley_x[i+1] = valley_x[i] - eta * .015 * (valley_x[i] - 1.4)
    valley_y[i+1] = (1-eta) * valley_y[i] - eta * noise
fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.55), sharex=True,
                         layout="constrained")
fig.suptitle("同一次训练中，进展与震荡可以并存", fontsize=21, fontweight="bold")
for ax in axes:
    ax.axvspan(cooling, end, color="#e1ecf7", alpha=.60)
    grid(ax)
axes[0].plot(steps, valley_y, color=orange, linewidth=2.2)
axes[0].axhline(0, color=dim, linewidth=1.4, linestyle=":")
axes[1].plot(steps, valley_x, color=teal, linewidth=3)
axes[0].set(ylabel="横向偏移 $y$")
axes[1].set(xlabel="训练更新步数", ylabel="沿谷底位置 $x$")
axes[0].text(94, .82, "学习率衰减期", color=blue, ha="center",
             bbox={"facecolor":paper, "edgecolor":"none", "pad":3})
axes[1].annotate("沿低曲率方向前进", xy=(39, valley_x[39]), xytext=(55, .36),
                 color=teal, arrowprops={"arrowstyle":"->","color":teal,"lw":1.6})
save(fig, "ch06-valley-trajectory")

# Ch06. One training backbone can yield multiple cooled endpoints.
fig, ax = plt.subplots(figsize=(7.6, 4.35), layout="constrained")
grid(ax)
ax.plot([0, 8, 91], [0, 1, 1], color=blue, linewidth=4.2,
        markevery=[1], marker="o", label="共享主干")
for start, stop, c in [(33, 48, orange), (57, 72, teal), (81, 96, "#754a91")]:
    ax.plot([start, stop], [1, 0], linestyle="--", color=c, linewidth=3.5)
    mark(ax, start, 1, c, size=65)
    mark(ax, stop, 0, c, "D", size=78)
    ax.text(stop, .13, f"预算 {1+(stop-48)//24}", ha="center", color=c)
ax.text(51, 1.10, "恒定学习率主干", color=blue, ha="center")
ax.set(title="WSD 可从同一主干切出多个退火分支",
       xlabel="训练进度（相对刻度）", ylabel="相对学习率",
       xlim=(0, 101), ylim=(-.08, 1.27), yticks=[0, .5, 1])
save(fig, "ch06-wsd-branches")

# Ch06. A shallow saddle can look stationary at first.
steps = np.arange(81)
saddle_eta, epsilon = .35, .18
saddle_x = .7 * (1-saddle_eta)**steps
saddle_y = .006 * (1+saddle_eta*epsilon)**steps
fig, ax = plt.subplots(figsize=(7.6, 4.3), layout="constrained")
ax.plot(steps, saddle_x, color=blue, linewidth=3.3,
        label=r"正曲率方向 $|x_t|$", marker="o", markevery=16)
ax.plot(steps, saddle_y, color=orange, linewidth=3.3, linestyle="--",
        label=r"负曲率方向 $|y_t|$", marker="s", markevery=16)
ax.axvspan(8, 40, color="#e6eee7", alpha=.72)
ax.text(25, .32, "两个偏移一度都很小", ha="center", color=teal)
ax.set(title="小负曲率与小初始偏移会延缓离开鞍点",
       xlabel="梯度下降步数", ylabel="坐标偏移幅度", xlim=(0, 80), ylim=(0, .88))
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncols=2)
grid(ax)
save(fig, "ch06-saddle-escape")

# Ch07. Exact iterates on a fixed one-dimensional quadratic.
fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
steps = np.arange(13)
for eta_h, col, label, line, marker in [
    (1.8, teal, "交替收敛", "-", "o"),
    (2, blue, "等幅震荡", ":", "s"),
    (2.2, orange, "交替发散", "--", "d"),
]:
    ax.plot(steps, (1-eta_h)**steps, linewidth=3, markersize=5.5,
            color=col, linestyle=line, marker=marker,
            label=fr"$\eta\lambda={eta_h:g}$  {label}")
ax.axhline(0, color=dim, linewidth=1.1, linestyle=":")
ax.set(title="固定曲率下，越过阈值会放大振幅",
       xlabel="梯度下降步数", ylabel="参数 $x_t$", xlim=(0, 12), xticks=[0, 3, 6, 9, 12])
grid(ax)
ax.legend(loc="upper left")
save(fig, "ch07-quadratic-threshold")

# Ch07. Exact local projection of (-g) at a hypothetical stability boundary.
fig, ax, x, y = contour_plot((-2.15, 2.35), (-1.55, 1.76), height=4.95)
ax.axvspan(-2.15, 0, color="#e1efe7", alpha=.88)
ax.axvline(0, color=teal, linewidth=2.6, linestyle="--")
ax.set(ylabel="", xlabel="", xticks=[], yticks=[])
for spine in ["left", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.text(-1.98, 1.47, "局部稳定区", color=teal)
ax.text(.17, 1.47, "锐度继续增大", color=orange)
ax.text(.16, .94, r"边界 $S=2/\eta$", color=teal)
center_y = .62 - 2 * 1.47 * 1.9 / 1.36
# Its Hessian is diag(1 + 0.45*x, 0.5), so x = 0 is the boundary at η = 2.
loss = .5 * (x-1.9)**2 + .25 * (y-center_y)**2 + .075 * x**3
ax.contour(x, y, loss, levels=[1.3, 2.3, 3.3, 4.5, 5.8, 7.3, 9],
           colors="#b6c4c5", linewidths=1.3)
origin = (0, .62)
def arrow(ax, at, to, col, line="-"):
    ax.annotate("", xy=to, xytext=at, arrowprops={
        "arrowstyle":"-|>", "lw":3.3, "color":col, "linestyle":line})
arrow(ax, origin, (1.36, -.85), orange)
arrow(ax, (1.36, -.85), (0, -.85), blue)
arrow(ax, origin, (0, -.85), teal)
mark(ax, *origin, ink, size=62)
ax.text(.78, .32, r"原下降方向 $-\nabla L$", color=orange)
ax.text(.2, -1.17, "震荡产生的锐度修正", color=blue)
ax.text(-.16, -.46, "平均更新", color=teal, ha="right")
ax.set_title("Central Flow 沿局部稳定边界推进")
save(fig, "ch07-central-flow-projection")

# Ch07. Exact curvature under a function-preserving rescaling.
fig, ax = plt.subplots(figsize=(7.6, 4.35), layout="constrained")
a = np.geomspace(.1, 10, 250)
sharpness = a**2 + a**-2
ax.plot(a, sharpness, color=blue, linewidth=3.6)
for location, value, text_x, text_y, label in [
    (1, 2, .68, 4.3, r"$(a,b)=(1,1)$  曲率 2"),
    (10, 100.01, 3.0, 77, r"$(a,b)=(10,0.1)$  曲率 100.01"),
]:
    mark(ax, location, value, orange, "D")
    ax.annotate(label, xy=(location,value), xytext=(text_x,text_y),
                color=orange, ha="center",
                arrowprops={"arrowstyle":"->","color":orange,"lw":1.6})
ax.set(title="预测不变时，Hessian 锐度仍可变化",
       xlabel=r"参数 $a$（对数刻度），全程 $ab=1$",
       ylabel=r"$\lambda_{\max}=a^2+a^{-2}$",
       xlim=(.089, 12), ylim=(1.5, 150), xscale="log", yscale="log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
ax.yaxis.set_major_locator(ticker.FixedLocator([2, 10, 25, 100]))
grid(ax)
save(fig, "ch07-rescale-sharpness")

# Ch07. Two one-dimensional neighborhoods with the same perturbation radius.
fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
rho = .35
delta = np.linspace(-.49, .49, 120)
ax.axvspan(-rho, rho, facecolor="#e9eee3")
ax.plot(delta, 2.6*delta**2, color=orange, linewidth=3.4, label="尖锐邻域")
ax.plot(delta, .11+.45*delta**2, color=teal, linewidth=3.4,
        linestyle="--", label="平坦邻域")
mark(ax, 0, 0, orange)
mark(ax, 0, .11, teal)
mark(ax, rho, 2.6*rho**2, orange, "D")
mark(ax, rho, .11+.45*rho**2, teal, "D")
ax.text(0, .38, r"同一扰动范围 $[-\rho,\rho]$", ha="center", color=dim)
ax.text(.36, .33, "最坏损失 0.319", color=orange, ha="right")
ax.text(.35, .184, "最坏损失 0.165", color=teal, ha="right")
ax.set(title="最坏邻域损失可以改变候选点的排序",
       xlabel=r"相对各自中心的参数扰动 $\delta$", ylabel="损失",
       xlim=(-.51,.51), ylim=(-.018,.58), xticks=[-.35, 0, .35])
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncols=2)
grid(ax)
save(fig, "ch07-sam-neighborhood")
