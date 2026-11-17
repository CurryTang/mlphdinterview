"""Generate original CS 312 foundations figures. Requires NumPy, Matplotlib and drawio."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, INK = "#1559b8", "#b55411", "#00816a", "#1b3047"
PALE = "#f4f7fb"
for candidate in ("Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS"):
    try:
        font_manager.findfont(candidate, fallback_to_default=False)
    except ValueError:
        continue
    plt.rcParams["font.family"] = ["DejaVu Sans", candidate]
    break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#a9bacc",
    "axes.axisbelow": True,
})


def plot(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.3, 4.7), layout="constrained")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=18, pad=12)
    ax.grid(alpha=0.35, color="#b7c7d8")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=155, facecolor="white")
    plt.close(fig)


def drawio_diagram(name, boxes, connections):
    """Render editable draw.io PNG using a cropped box-and-arrow layout."""
    document = ET.Element("mxfile")
    page = ET.SubElement(document, "diagram", name="Diagram")
    graph = ET.SubElement(page, "mxGraphModel", background="#ffffff", grid="0")
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    for key, label, x, y, width, height, color in boxes:
        card = ET.SubElement(
            root, "mxCell", id=key, value=label, vertex="1", parent="1",
            style=(f"rounded=1;arcSize=14;whiteSpace=wrap;html=1;fontSize=17;"
                   f"fontColor={INK};strokeColor={color};strokeWidth=1.5;fillColor={PALE};"),
        )
        ET.SubElement(
            card, "mxGeometry", x=str(x), y=str(y),
            width=str(width), height=str(height), attrib={"as": "geometry"},
        )
    for index, (start, end, label, exit_x, exit_y, entry_x, entry_y) in enumerate(connections):
        arrow = ET.SubElement(
            root, "mxCell", id=f"e{index}", value=label, edge="1", parent="1",
            source=start, target=end,
            style=(f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                   f"strokeWidth=2;strokeColor=#4e6680;fontSize=14;"
                   f"exitX={exit_x};exitY={exit_y};entryX={entry_x};entryY={entry_y};"),
        )
        ET.SubElement(arrow, "mxGeometry", relative="1", attrib={"as": "geometry"})
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / f"{name}.drawio"
        ET.ElementTree(document).write(source, encoding="utf-8", xml_declaration=True)
        subprocess.run([
            "drawio", "-x", "-f", "png", "-e", "-b", "24", "-s", "2",
            "-o", str(ASSETS / f"{name}.drawio.png"), str(source),
        ], check=True, timeout=45)


def foundations():
    fig, ax = plot("正确 token 的概率与损失", "模型给正确 token 的概率", "单 token 损失（nats）")
    p = np.linspace(0.02, 1, 220)
    ax.plot(p, -np.log(p), lw=3, color=BLUE)
    ax.scatter([0.1, 0.2], -np.log([0.1, 0.2]), s=78, color=ORANGE, zorder=5)
    ax.annotate("概率从 0.1 到 0.2\n损失降低 ln 2",
                xy=(0.2, -np.log(0.2)), xytext=(0.37, 2.55),
                arrowprops={"arrowstyle": "->", "color": INK})
    save(fig, "ch01-token-probability")

    fig, ax = plot("同配方比较与分别调参", "峰值 LR（示意，对数轴）", "最终验证损失（示意）")
    x = np.geomspace(.0005, .024, 160)
    for label, optimum, floor, color in (
        ("基线", .003, 2.94, BLUE), ("结构消融", .001, 3.02, ORANGE),
    ):
        ax.semilogx(x, floor + .14*np.log10(x/optimum)**2, lw=3, c=color, label=label)
        ax.scatter(optimum, floor, c=color, s=65, zorder=5)
    ax.axvline(.003, ls="--", color=GREEN, lw=2, label="共同配方")
    ax.legend(loc="upper left", fontsize=13)
    save(fig, "ch01-ablation-sweep")

    fig, ax = plot("因果掩码与目标移位", "键位置（可供读取）", "查询位置（产生预测）")
    attention = np.tril(np.ones((4, 4)))
    ax.imshow(attention, cmap=ListedColormap(["#e9eef5", "#b5d8fd"]), aspect="auto")
    ax.set_xticks(range(4), ["BOS", "深", "度", "学"])
    ax.set_yticks(range(4), ["BOS → 深", "深 → 度", "度 → 学", "学 → ·"])
    ax.grid(False)
    for row, col in np.ndindex(attention.shape):
        ax.text(col, row, "可见" if attention[row, col] else "屏蔽",
                ha="center", va="center", fontsize=14, color=INK)
    save(fig, "ch02-causal-mask")

    fig, ax = plot("RoPE 中的位置差决定相对角度", "位置 u（固定查询位置 t = 2）", "旋转后查询与键的点积")
    position = np.arange(17)
    cosine = np.cos((position - 2) * np.pi / 6)
    ax.plot(position, cosine, "o-", lw=2.8, color=BLUE)
    ax.scatter([0, 4], cosine[[0, 4]], color=ORANGE, s=85, zorder=5)
    ax.annotate("相对距离 ±2 给出相同点积\n此例使用相同的初始单位向量",
                xy=(4, cosine[4]), xytext=(6.2, .78),
                arrowprops={"arrowstyle": "->", "color": INK}, fontsize=13)
    ax.set_xticks(np.arange(0, 17, 2))
    save(fig, "ch02-rope-relative-position")

    fig, ax = plot("用留出的规模检验外推误差", "参数规模（相对单位，对数轴）", "验证损失（合成示例）")
    sizes = np.geomspace(1, 64, 120)
    cur = 2.9 + .75*sizes**(-.36)
    observed = cur + np.clip(np.log2(sizes)-3, 0, None)**2*.025
    ax.semilogx(sizes, cur, lw=2.8, c=BLUE, ls="--", label="小尺度拟合")
    ax.semilogx(sizes, observed, lw=2.8, c=ORANGE, label="含规模变化的结果")
    x = 2 ** np.arange(7)
    ax.scatter(x[:4], (2.9 + .75*x[:4]**(-.36)), c=BLUE, s=46, zorder=5)
    ax.axvspan(8, 64, color=BLUE, alpha=.08, label="外推区间")
    ax.set_xticks(x, [str(v) for v in x])
    ax.legend(fontsize=12)
    save(fig, "ch03-scaling-extrapolation")

    fig, ax = plot("批均值噪声随 batch 增大而下降", "batch 大小（对数轴）", "逐步损失波动的标准差")
    batch = np.array([8, 16, 32, 64, 128, 256, 512])
    ax.loglog(batch, .04*np.sqrt(64/batch), "o-", color=BLUE, lw=3, label="独立样本假设")
    for x, y in [(16, .08), (64, .04), (256, .02)]:
        ax.annotate(f"{y:.2f}", (x, y), xytext=(5, 9), textcoords="offset points", color=ORANGE)
    ax.set_xticks(batch, [str(value) for value in batch])
    ax.set_yticks([.01, .02, .04, .08, .16], [.01, .02, .04, .08, .16])
    ax.legend(fontsize=12)
    save(fig, "ch03-batch-jitter")

    fig, ax = plot("相同峰值 LR 下的四种调度", "训练进度（%）", "学习率 / 峰值学习率")
    progress = np.linspace(0, 1, 301)
    warmup = 0.05
    after = np.clip((progress-warmup)/(1-warmup), 0, 1)
    for label, y, color, style in [
        ("线性", 1-after, BLUE, "-"),
        ("余弦", .5+.5*np.cos(np.pi*after), ORANGE, "--"),
        ("WSD（末段 20%）", np.clip((1-progress)/.2, 0, 1), GREEN, "-."),
    ]:
        ax.plot(100*progress, np.minimum(progress/warmup, y), color=color, lw=2.6, ls=style, label=label)
    ax.plot([0,100], [1,1], c="#8353ad", lw=2.6, ls=":", label="恒定")
    ax.axvspan(0, 5, color=BLUE, alpha=.10)
    ax.legend(ncols=2, fontsize=12, loc="lower left")
    save(fig, "ch03-lr-schedules")

    fig, ax = plot("二次模型的收缩速度与稳态噪声", "更新步数", "期望方向损失（对数轴）")
    t = np.arange(19)
    for lr, batch, color in [(.1, 16, BLUE), (.2, 32, ORANGE), (.45, 72, GREEN)]:
        a = 1-4*lr
        floor = lr*2/(2*batch*(2-4*lr))
        ax.semilogy(t, a**(2*t) + (1-a**(2*t))*floor,
                    color=color, lw=3, label=f"LR={lr:g}, B={batch}")
        ax.axhline(floor, ls=":", c=color, alpha=.6)
    ax.legend(fontsize=12)
    save(fig, "ch04-nqm-risk")

    fig, ax = plot("动量在同一曲率方向上的阻尼", "更新步数", "参数坐标（初值为 1）")
    mu, curvature = .81, 1
    for lr, label, color in [
        (.003, "小步长：过阻尼", BLUE),
        ((1-np.sqrt(mu))**2, "临界阻尼", GREEN),
        (.07, "较大步长：欠阻尼", ORANGE),
    ]:
        previous = current = 1.
        trajectory = [current]
        for _ in range(65):
            previous, current = current, (1+mu-lr*curvature)*current-mu*previous
            trajectory.append(current)
        ax.plot(trajectory, c=color, lw=2.8, label=label)
    ax.axhline(0, color="#8296ac", lw=1)
    ax.legend(fontsize=12)
    save(fig, "ch04-momentum-damping")

    fig, ax = plot("保持 LR × WD 时的近似角度", "峰值 LR（对数轴）", "权重衰减 WD（对数轴）")
    lr, wd = np.geomspace(.0003, .012, 120), np.geomspace(.035, 2, 120)
    x,y = np.meshgrid(lr, wd)
    angle = np.sqrt(2*x*y)
    ax.set(xscale="log", yscale="log")
    cs = ax.contour(x, y, angle, levels=[.012, .025, np.sqrt(.0012), .05, .10], colors=BLUE)
    ax.clabel(cs, fmt=lambda value:f"{value:.3f} rad", fontsize=11)
    ax.scatter([.0015,.003,.006], [.4,.2,.1], c=ORANGE, s=75, zorder=5)
    ax.set_xticks([.0003,.001,.003,.01], ["0.0003","0.001","0.003","0.01"])
    ax.set_yticks([.05,.1,.2,.4,1,2], ["0.05","0.1","0.2","0.4","1","2"])
    save(fig, "ch04-lr-wd-angle")

    fig, ax = plot("Hyperball 把试探点投影回固定半径", "参数坐标 1", "参数坐标 2")
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#a3b4c7", lw=2, ls="--")
    start, tentative = np.array([1., 0.]), np.array([1.10, .67])
    end = tentative/np.linalg.norm(tentative)
    for a, b, color in [(start, tentative, ORANGE), (tentative, end, GREEN)]:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 3})
    ax.scatter(*start, color=BLUE, s=60, zorder=5)
    ax.scatter(*end, color=GREEN, s=60, zorder=5)
    ax.text(.43, -.19, "当前权重", color=BLUE)
    ax.text(1.19, .53, "试探更新", color=ORANGE, fontsize=13)
    ax.text(.53, .78, "投影后的权重", color=GREEN, fontsize=13)
    ax.set(xlim=(-1.28, 1.95), ylim=(-1.18, 1.14), aspect="equal")
    save(fig, "ch04-hyperball-projection")

    fig, ax = plot("宽度增加时，累加尺度取决于对齐", "宽度 n（对数轴）", "对齐比 R（对数轴）")
    widths = 2 ** np.arange(4, 12)
    for exponent, label, color in [
        (.5, "独立累加：n 的 1/2 次方", BLUE),
        (.75, "部分对齐：n 的 3/4 次方", ORANGE),
        (1, "完全对齐：n 的 1 次方", GREEN),
    ]:
        ax.loglog(widths, widths**exponent, "o-", lw=3, c=color, label=label)
    ax.set_xticks(widths[::2], [str(v) for v in widths[::2]])
    ax.legend(fontsize=12, loc="upper left")
    save(fig, "ch05-width-alignment")

    fig, ax = plot("学习率在不同宽度间的转移", "宽度 / 基准宽度（对数轴）", "各处方的最优基准 LR 比（示意）")
    ratio = 2.0**np.arange(-2, 3)
    for power, label, color in [
        (-1, "最优 LR 随宽度逆比下降", BLUE),
        (-.56, "中间处方（指数 -0.56）", ORANGE),
        (0, "共享基准 LR 近似不变", GREEN),
    ]:
        ax.loglog(ratio, ratio**power, "o-", lw=3, c=color, label=label)
    ax.set_xticks(ratio, [str(v) for v in ratio])
    ax.set_yticks(ratio, [str(v) for v in ratio])
    ax.legend(fontsize=12)
    save(fig, "ch05-learning-rate-transfer")

    fig, ax = plot("谱约束改变各奇异方向的相对更新幅度", "梯度的奇异方向", "更新奇异值（峰值归一化）")
    ind = np.arange(3)
    ax.bar(ind-.17, [1, 1/9, 0], width=.34, color=BLUE, label="原方向")
    ax.bar(ind+.17, [1, 1, 0], width=.34, color=ORANGE, label="理想谱约束")
    ax.set_xticks(ind, ["σ = 9", "σ = 1", "σ = 0"])
    ax.set_ylim(0, 1.23)
    ax.legend(fontsize=12, loc="upper right")
    save(fig, "ch05-muon-singular-spectrum")


    fig, ax = plot("深度缩放取决于残差方向是否一致", "残差分支数 L（对数轴）", "累积残差量级（对数轴）")
    depth = 2 ** np.arange(9)
    ax.loglog(depth, np.sqrt(depth), "o-", c=BLUE, lw=2.8, label="独立分支：√L")
    ax.loglog(depth, depth, "o-", c=ORANGE, lw=2.8, label="同向分支：L")
    ax.loglog(depth, np.ones(len(depth)), "o--", c=GREEN, lw=2.6,
              label="各自乘对应倒数后")
    ax.set_xticks(depth[::2], [str(value) for value in depth[::2]])
    ax.legend(fontsize=12)
    save(fig, "ch05-depth-residual-scaling")


def diagrams():
    drawio_diagram("ch01-experiment-loop", [
        ("a", "提出假设<br>写下排序和数量级", 0, 0, 165, 76, BLUE),
        ("b", "设计对照<br>固定预算和测量口径", 230, 0, 165, 76, BLUE),
        ("c", "运行实验<br>保存曲线和种子", 230, 141, 165, 76, BLUE),
        ("d", "检验预测<br>用证据更新机制解释", 0, 141, 165, 76, GREEN),
    ], [
        ("a", "b", "", 1, .5, 0, .5),
        ("b", "c", "", .5, 1, .5, 0),
        ("c", "d", "", 0, .5, 1, .5),
        ("d", "a", "形成下一问", .5, 0, .5, 1),
    ])
    drawio_diagram("ch02-transformer-flow", [
        ("a", "输入 token ID<br>B × T", 0, 0, 160, 62, BLUE),
        ("b", "Embedding 查表<br>B × T × d", 225, 0, 182, 62, BLUE),
        ("c", "Pre-norm 块 × L<br>注意力残差<br>再接 MLP 残差", 225, 111, 182, 82, GREEN),
        ("d", "最终 RMSNorm<br>B × T × d", 0, 121, 160, 62, BLUE),
        ("e", "词表投影<br>logits：B × T × V", 0, 236, 160, 62, BLUE),
        ("f", "移位目标、交叉熵<br>平均到有效 token", 225, 236, 182, 62, ORANGE),
    ], [
        ("a", "b", "", 1, .5, 0, .5),
        ("b", "c", "", .5, 1, .5, 0),
        ("c", "d", "", 0, .5, 1, .5),
        ("d", "e", "", .5, 1, .5, 0),
        ("e", "f", "", 1, .5, 0, .5),
    ])


if __name__ == "__main__":
    foundations()
    if shutil.which("drawio"):
        diagrams()
