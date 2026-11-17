"""Render the worked examples in chapter 11 with NumPy and Matplotlib."""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
NAVY = "#17263d"
BLUE = "#285cad"
ORANGE = "#b65320"
TEAL = "#14766c"
PURPLE = "#6650a4"
SLATE = "#64748b"

font_path = Path(os.getenv("CS312_CJK_FONT", "/System/Library/Fonts/Hiragino Sans GB.ttc"))
if font_path.is_file():
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = [family, "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9ba8ba",
    "axes.labelcolor": NAVY,
    "axes.labelsize": 12,
    "axes.labelpad": 8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "font.size": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "mathtext.fontset": "stix",
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
})


def figure(x_label, y_label):
    fig, ax = plt.subplots(figsize=(6.4, 3.75), layout="constrained")
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid(axis="y", color="#d4dce8", linewidth=0.7, alpha=0.8)
    return fig, ax


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / f"ch11-{name}.png", dpi=200, pad_inches=0.10)
    plt.close(fig)


def double_descent():
    fig, ax = figure("参数量 / 样本量  γ", "预测风险")
    left = np.linspace(0.03, 0.98, 200)
    right = np.linspace(1.02, 4.0, 500)
    bias = 1 - 1 / right
    variance_above = 0.24 / (right - 1)
    variance_below = 0.24 * left / (1 - left)
    ax.plot(left, variance_below, color=BLUE, label="总风险")
    ax.plot(right, bias + variance_above, color=BLUE)
    ax.plot(right, variance_above, color=ORANGE, linestyle="--", label="方差")
    ax.plot(right, bias, color=TEAL, linestyle="-.", label="偏差平方")
    ax.axvline(1, color=SLATE, linestyle=":", linewidth=1.5)
    ax.annotate("插值门槛", (1, 2.14), xytext=(0.14, 2.14),
                arrowprops={"arrowstyle": "->", "color": SLATE})
    ax.axhline(1, color=SLATE, alpha=0.55, linewidth=1, linestyle=":")
    ax.text(3.95, 1.05, "全零预测器", ha="right", color=SLATE, fontsize=9)
    ax.set(xlim=(0, 4), ylim=(0, 2.5), xticks=[0, 1, 2, 3, 4])
    ax.legend(loc="upper right", ncols=1)
    save(fig, "double-descent")


def repeated_epochs():
    fig, ax = figure("同一语料的遍历轮数", "损失（示意单位）")
    epoch = np.arange(1, 33)
    train = 1.95 + 1.75 * np.exp(-epoch / 7)
    val = 2.40 + 1.35 * np.exp(-epoch / 4.5) + 0.023 * epoch
    best = np.argmin(val)
    ax.plot(epoch, train, color=BLUE, label="训练集")
    ax.plot(epoch, val, color=ORANGE, linestyle="--", label="验证集")
    ax.axvline(epoch[best], color=SLATE, linestyle=":", linewidth=1.3)
    ax.scatter(epoch[best], val[best], s=48, color=ORANGE, edgecolor="white", zorder=3)
    ax.annotate("验证损失最低", (epoch[best], val[best]), xytext=(16, 2.30),
                arrowprops={"arrowstyle": "->", "color": SLATE})
    ax.legend(loc="upper right")
    ax.set(xlim=(1, 32), xticks=[1, 8, 16, 24, 32], ylim=(1.8, 3.75))
    save(fig, "repeated-epochs")


def ensemble_correlation():
    fig, ax = figure("集成成员数  K", "平均误差方差 / 单成员方差")
    members = np.arange(1, 17)
    for rho, color, style, mark in [
        (0.0, BLUE, "-", "o"), (0.3, TEAL, "--", "s"), (0.8, ORANGE, "-.", "D")
    ]:
        ax.plot(members, rho + (1 - rho) / members, color=color, linestyle=style,
                marker=mark, markevery=[0, 1, 3, 7, 15], markersize=4,
                label=rf"$\rho={rho:g}$")
        if rho:
            ax.axhline(rho, color=color, linewidth=1.1, alpha=0.50, linestyle=":")
    ax.legend(loc="center right")
    ax.set(xlim=(0.8, 16.25), ylim=(0, 1.05), xticks=[1, 2, 4, 8, 12, 16])
    save(fig, "ensemble-correlation")


def distillation_results():
    fig, ax = figure("验证交叉熵（越低越好）", "")
    labels = ["300M 正则化单模型", "300M 蒸馏学生", "8 × 300M 集成教师"]
    scores = [3.57, 3.36, 3.32]
    colors = [SLATE, TEAL, BLUE]
    for y, score, color in zip(range(3), scores, colors):
        ax.hlines(y, 3.25, score, color=color, alpha=0.36)
        ax.scatter(score, y, color=color, s=76, zorder=3)
        ax.text(score + 0.011, y, f"{score:.2f}", va="center", color=color)
    ax.set(xlim=(3.25, 3.66), ylim=(-0.55, 2.55), yticks=range(3),
           yticklabels=labels, xticks=[3.30, 3.40, 3.50, 3.60])
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#d4dce8", linewidth=0.7, alpha=0.8)
    save(fig, "distillation-results")


def transfer_equivalence():
    fig, ax = figure("目标域训练数据（MB）", "目标域验证损失（示意单位）")
    amount = np.geomspace(3, 150, 240)
    at_target = 2.5 + 5 / np.sqrt(50)
    ax.plot(amount, 2.5 + 5 / np.sqrt(amount), color=BLUE, label="从头训练")
    ax.plot(amount, 2.5 + 5 / np.sqrt(amount + 40), color=PURPLE,
            linestyle="--", label="预训练后微调")
    ax.vlines([10, 50], ymin=2.60, ymax=at_target, color=SLATE, linestyle=":", linewidth=1.4)
    ax.hlines(at_target, xmin=3, xmax=50, color=SLATE, linestyle=":", linewidth=1.4)
    ax.scatter([10, 50], [at_target, at_target], color=[PURPLE, BLUE], s=48, zorder=3)
    ax.text(12, at_target + 0.11, "相同损失", fontsize=10)
    ax.set(xscale="log", xlim=(3, 150), ylim=(2.6, 5.5), xticks=[5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.legend(loc="upper right")
    save(fig, "transfer-equivalence")


if __name__ == "__main__":
    double_descent()
    repeated_epochs()
    ensemble_correlation()
    distillation_results()
    transfer_equivalence()
