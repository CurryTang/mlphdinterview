"""Render the formula-derived PNGs for the three architecture chapters."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ASSETS = Path(__file__).resolve().parents[1] / "assets"
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]
for font in FONT_CANDIDATES:
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=font
        ).get_name()
        break
else:
    raise SystemExit("Install Noto Sans CJK at the listed Linux path to render Chinese labels.")

INK = "#111827"
SLATE = "#475569"
BLUE = "#4338ca"
ORANGE = "#b45309"
TEAL = "#0f766e"
PINK = "#be185d"
PAPER = "#f8fafc"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": "#cbd5e1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e2e8f0",
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": PAPER,
    }
)


def chart(title, x_label, y_label):
    figure, axis = plt.subplots(figsize=(7.1, 4.6), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    axis.set_title(title, loc="left", fontsize=20, fontweight="bold", pad=15)
    axis.set(xlabel=x_label, ylabel=y_label)
    axis.set_axisbelow(True)
    axis.grid(axis="y")
    return figure, axis


def line(axis, x, y, color, label, style="-"):
    axis.plot(x, y, linestyle=style, color=color, lw=2.8, label=label)


def save(figure, name):
    figure.savefig(ASSETS / f"{name}.png", dpi=150)
    plt.close(figure)


def legend(axis, **options):
    axis.legend(facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=12, **options)


def depth():
    layers = np.arange(61)
    fig, ax = chart("临界点附近的相关差异衰减更慢", "层数", "剩余差异 / 初始差异")
    for chi, color, style in [(0.9, ORANGE, "--"), (0.95, BLUE, "-"), (0.99, TEAL, "-.")]:
        line(ax, layers, chi**layers, color, f"χ = {chi:g}", style)
    ax.axhline(math.exp(-1), color="#94a3b8", linestyle=":")
    ax.text(61, math.exp(-1), r"$1/e$", va="center", ha="right", fontsize=11, color=SLATE)
    legend(ax, loc="upper right")
    save(fig, "ch08-correlation")

    fig, ax = chart("残差方差取决于分支如何缩放", "层数", "方差倍数（对数轴）")
    layers = np.arange(17)
    line(ax, layers, 2.0**layers, ORANGE, "分支方差 = 当前流方差", "--")
    line(ax, layers, 1 + layers, BLUE, "分支增加固定方差")
    line(ax, layers, (1 + 1 / 16) ** layers, TEAL, "分支方差 = 当前方差 / 16", "-.")
    ax.set_yscale("log")
    legend(ax, loc="upper left")
    save(fig, "ch08-variance")

    fig, ax = chart("混合矩阵保留均值，衰减流间差异", "混合次数", "剩余范数 / 初始范数")
    steps = np.arange(11)
    line(ax, steps, np.ones_like(steps), BLUE, "同向分量 · 特征值 1")
    line(ax, steps, 0.6**steps, ORANGE, "反向分量 · 特征值 0.6", "--")
    ax.set_xticks(np.arange(0, 11, 2))
    legend(ax)
    save(fig, "ch08-stream-mixing")


def sequence():
    distance = np.arange(51)
    fig, ax = chart("全序列梯度会混入各位置的近程误差", "距最后位置的步数", "梯度幅度（对数轴）")
    last_only = 0.8**distance
    all_positions = (1 - 0.8 ** (distance + 1)) / (0.2 * 100)
    line(ax, distance, last_only, BLUE, "末位损失")
    line(ax, distance, all_positions, ORANGE, "全序列平均损失", "--")
    ax.set_yscale("log")
    ax.set_yticks([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda magnitude, _: f"{magnitude:.5f}".rstrip("0").rstrip(".")))
    legend(ax)
    save(fig, "ch09-time-gradient")

    fig, ax = chart("近零 value 可接收大部分注意力权重", "可读位置", "归一化权重")
    positions = np.arange(3)
    values = np.array([1.0, 1.0, 0.0])
    scores = np.array([0.0, 0.0, math.log(18)])
    weights = np.exp(scores) / np.exp(scores).sum()
    alternative = np.array([0.5, 0.5, 0.0])
    width = 0.34
    ax.bar(positions - width / 2, alternative, width, color=BLUE,
           label=f"移除 sink · 输出 {alternative @ values:.1f}")
    ax.bar(positions + width / 2, weights, width, color=ORANGE,
           hatch="//", label=f"保留 sink · 输出 {weights @ values:.1f}")
    ax.set_xticks(positions, ["位置 A", "位置 B", "sink"])
    ax.set_ylim(0, 1.03)
    legend(ax, loc="upper left")
    save(fig, "ch09-sink")

    fig, ax = chart("重复 key 时，三种状态更新给出不同读数", "已写入次数", "同一 key 的读出值")
    targets = [2, 5, 1, 4]
    events = np.arange(len(targets) + 1)
    for beta, color, style in [(1.0, TEAL, "-."), (0.5, ORANGE, "--")]:
        reads = [0.0]
        for target in targets:
            reads.append((1 - beta) * reads[-1] + beta * target)
        line(ax, events, reads, color, f"DeltaNet · β={beta:g}", style)
    line(ax, events, np.cumsum([0, *targets]), BLUE, "线性注意力 · 累加")
    ax.set_xticks(events, ["初始", *(str(value) for value in targets)])
    ax.set_xlabel("同一 key 依次写入的目标值")
    legend(ax, loc="upper left")
    save(fig, "ch09-delta-write")

    fig, ax = chart("矩阵状态大小与上下文长度无关", "上下文长度（token）", "每头存储量（标量数）")
    lengths = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    line(ax, lengths, 2 * 64 * lengths, BLUE, "完整 KV 缓存 · 2 × 64 × T")
    line(ax, lengths, np.full_like(lengths, 64**2), ORANGE, "矩阵状态 · 64 × 64", "--")
    ax.scatter(32, 4096, s=70, color=TEAL, zorder=5)
    ax.annotate("32 token 处相交", (32, 4096), xytext=(55, 650),
                arrowprops={"arrowstyle": "->", "color": SLATE}, fontsize=12)
    ax.set(xscale="log", yscale="log", ylim=(450, 9e5))
    ax.set_xticks([16, 64, 256, 1024, 4096], ["16", "64", "256", "1024", "4096"])
    legend(ax, loc="upper left")
    save(fig, "ch09-memory")


def capacity():
    fig, ax = chart("重复折叠会把线性区间数翻倍", "输入 x", "输出")
    x = np.linspace(0, 1, 1001)
    folded = x
    for repetition, color, style in [(1, BLUE, "-"), (2, ORANGE, "--"), (3, TEAL, "-.")]:
        folded = np.minimum(2 * folded, 2 - 2 * folded)
        line(ax, x, folded, color, f"折叠 {repetition} 次 · {2**repetition} 段", style)
    ax.set(ylim=(-0.03, 1.42), yticks=[0, 0.25, 0.5, 0.75, 1])
    legend(ax, loc="upper right", ncols=3, columnspacing=0.8)
    save(fig, "ch10-folds")

    fig, ax = chart("缩窄输出头会减少梯度可见方向", "读出列空间的秩", "投影外能量（%）")
    ranks = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
    outside = 100 * (1 - ranks / 4096)
    line(ax, ranks, outside, BLUE, "各向同性参考 · 词表 4096")
    ax.scatter(ranks, outside, color=BLUE, zorder=4)
    ax.annotate("秩 128 · 96.875%", (128, 96.875), xytext=(27, 70),
                arrowprops={"arrowstyle": "->", "color": SLATE}, fontsize=12)
    ax.set(xscale="log", ylim=(-3, 107))
    ax.set_xticks([32, 128, 512, 2048, 4096], ["32", "128", "512", "2048", "4096"])
    legend(ax, loc="lower left")
    save(fig, "ch10-gradient-rank")

    fig, ax = chart("共享方式改变参数量与调用次数的关系", "核心结构", "FFN 或核心块份数")
    choices = np.arange(4)
    width = 0.34
    ax.bar(choices - width / 2, [6, 1, 4, 8], width, color=BLUE, label="独立权重份数")
    ax.bar(choices + width / 2, [6, 6, 1, 2], width, color=ORANGE,
           hatch="//", label="对单 token 的调用")
    ax.set_xticks(choices, ["六个独立块", "单块循环六次", "4 专家 top-1", "8 专家 top-2"])
    ax.set(ylim=(0, 9), yticks=range(0, 9, 2))
    legend(ax, loc="upper left")
    save(fig, "ch10-parameters")

    counts = np.arange(101)
    choose = np.array([math.comb(100, int(count)) for count in counts], dtype=float)
    fig, ax = chart("长度不变也会出现计数分布偏移", "100 个 token 中 1 的个数", "概率")
    for rate, color, style in [(0.5, BLUE, "-"), (0.9, PINK, "--")]:
        mass = choose * rate**counts * (1 - rate) ** (100 - counts)
        line(ax, counts, mass, color, f"每位取 1 的概率 {rate:g}", style)
    ax.set(xlim=(24, 101), ylim=(0, 0.145))
    legend(ax, loc="upper left")
    save(fig, "ch10-parity-shift")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    depth()
    sequence()
    capacity()
