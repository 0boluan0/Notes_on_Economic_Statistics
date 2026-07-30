#!/usr/bin/env python3
"""Generate deterministic teaching figures for MIT 6.042J notes."""

from __future__ import annotations

from math import comb, gcd, pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "98_attachment" / "mathematics_for_computer_science" / "mit6_042j"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#16A34A"
RED = "#DC2626"
PURPLE = "#7C3AED"
INK = "#263238"
GRAY = "#94A3B8"
LIGHT = "#E2E8F0"
PALE_BLUE = "#DBEAFE"
PALE_GREEN = "#DCFCE7"
PALE_RED = "#FEE2E2"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "dejavusans",
    }
)


def canvas(title: str, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, x=0.04, y=0.97, ha="left", fontweight="bold", color=INK)
    if ncols == 1:
        axes = [axes]
    return fig, axes


def finish(fig, stem: str) -> None:
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.85, wspace=0.28)
    fig.savefig(OUT / f"{stem}.png", dpi=200, facecolor="white")
    plt.close(fig)


def clean(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def node(ax, xy, label, color=BLUE, radius=0.055, text_color="white"):
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor="white", lw=1.5, zorder=3))
    ax.text(*xy, label, ha="center", va="center", color=text_color, fontweight="bold", zorder=4)


def box(ax, xy, width, height, label, color=PALE_BLUE, edge=BLUE, size=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color, edgecolor=edge, lw=1.5
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size)


def arrow(ax, start, end, color=INK, text=None, bend=0.0, lw=1.7, shrink=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
        connectionstyle=f"arc3,rad={bend}", shrinkA=shrink, shrinkB=shrink, zorder=2
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + 0.04 + 0.12 * bend
        ax.text(mx, my, text, ha="center", va="center", fontsize=9, color=color)


def fig_implication():
    fig, (ax1, ax2) = canvas("Implication: only one row is false", 2)
    ax1.axis("off")
    rows = [["T", "T", "T"], ["T", "F", "F"], ["F", "T", "T"], ["F", "F", "T"]]
    table = ax1.table(cellText=rows, colLabels=["P", "Q", r"$P\Rightarrow Q$"], loc="center", cellLoc="center")
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(PALE_RED if r == 2 else (PALE_BLUE if r == 0 else "#F8FAFC"))
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax1.text(0.5, 0.10, "A true premise cannot lead to a false conclusion.", ha="center", color=RED, fontsize=9)
    clean(ax2)
    box(ax2, (0.08, 0.56), 0.28, 0.18, "assume P")
    box(ax2, (0.64, 0.56), 0.28, 0.18, "derive Q", PALE_GREEN, GREEN)
    arrow(ax2, (0.36, 0.65), (0.64, 0.65), ORANGE, "valid steps")
    box(ax2, (0.36, 0.18), 0.28, 0.16, r"prove $P\Rightarrow Q$", "#F3E8FF", PURPLE)
    arrow(ax2, (0.78, 0.56), (0.56, 0.34), GREEN)
    finish(fig, "unit01-proof-implication")


def fig_sets():
    fig, axes = canvas("Set operations are regions, not symbols", 2)
    for ax, mode in zip(axes, ["intersection", "difference"]):
        clean(ax, (0, 1.2), (0, 1))
        xx, yy = np.meshgrid(np.linspace(0, 1.2, 500), np.linspace(0, 1, 420))
        in_a = (xx - 0.48) ** 2 + (yy - 0.52) ** 2 <= 0.28**2
        in_b = (xx - 0.73) ** 2 + (yy - 0.52) ** 2 <= 0.28**2
        region = (in_a & in_b) if mode == "intersection" else (in_a & ~in_b)
        ax.contourf(xx, yy, region.astype(float), levels=[0.5, 1.5],
                    colors=[ORANGE if mode == "intersection" else RED], alpha=0.62)
        ax.add_patch(Circle((0.48, 0.52), 0.28, facecolor="none", edgecolor=BLUE, lw=2))
        ax.add_patch(Circle((0.73, 0.52), 0.28, facecolor="none", edgecolor=GREEN, lw=2))
        ax.text(0.35, 0.79, "A", color=BLUE, fontweight="bold")
        ax.text(0.84, 0.79, "B", color=GREEN, fontweight="bold")
        if mode == "intersection":
            ax.text(0.605, 0.18, r"$A\cap B$", ha="center", fontweight="bold")
        else:
            ax.text(0.60, 0.18, r"$A-B=A\cap B^c$", ha="center", fontweight="bold")
    finish(fig, "unit01-set-operations")


def fig_induction():
    fig, (ax,) = canvas("Induction: base case plus a reusable implication")
    clean(ax, (0, 10), (0, 4))
    angles = [18, 15, 12, 9, 6, 3, 0, 0, 0]
    for i, angle in enumerate(angles):
        color = GREEN if i == 0 else (ORANGE if i < 6 else BLUE)
        rect = Rectangle((0.7 + i, 1.15), 0.22, 1.45, angle=angle, facecolor=color, edgecolor="white", lw=1.2)
        ax.add_patch(rect)
        ax.text(0.82 + i, 0.78, f"P({i})", ha="center", fontsize=9)
    ax.text(0.7, 3.35, "Base", color=GREEN, fontweight="bold")
    ax.text(3.9, 3.35, r"$P(k)\Rightarrow P(k+1)$", color=ORANGE, fontweight="bold", ha="center")
    arrow(ax, (1.05, 3.0), (6.5, 3.0), ORANGE)
    ax.text(8.2, 3.35, "All cases", color=BLUE, fontweight="bold", ha="center")
    finish(fig, "unit01-induction-dominoes")


def fig_wop():
    fig, (ax,) = canvas("Well-ordering: a nonempty subset of N has a least element")
    clean(ax, (-0.5, 10.5), (-0.2, 3.2))
    xs = np.arange(10)
    ax.scatter(xs, np.ones_like(xs), s=110, color=LIGHT, edgecolor=INK, zorder=2)
    subset = [3, 5, 6, 9]
    ax.scatter(subset, np.ones(len(subset)), s=140, color=BLUE, edgecolor="white", zorder=3)
    ax.annotate("least element", xy=(3, 1.05), xytext=(1.8, 2.1), color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.text(4.5, 0.35, r"$S=\{3,5,6,9\}\subseteq\mathrm{N}$", ha="center")
    for x in [8.5, 7.3, 6.1, 4.9]:
        arrow(ax, (x, 2.75), (x - 0.9, 2.75), RED, lw=1.4)
    ax.text(6.7, 2.95, "an infinite descent would contradict WOP", ha="center", color=RED, fontsize=9)
    finish(fig, "unit01-well-ordering-descent")


def fig_invariant():
    fig, (ax,) = canvas("State-machine invariant: every reachable state stays in the safe region")
    clean(ax, (0, 1.4), (0, 1))
    safe = FancyBboxPatch((0.05, 0.12), 0.82, 0.75, boxstyle="round,pad=0.02", facecolor=PALE_GREEN, edgecolor=GREEN, lw=2)
    ax.add_patch(safe)
    ax.text(0.08, 0.82, r"Invariant $I(s)$", color=GREEN, fontweight="bold")
    pts = [(0.22, 0.58), (0.48, 0.67), (0.68, 0.43), (0.37, 0.30)]
    for i, p in enumerate(pts):
        node(ax, p, str(i), GREEN)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]:
        arrow(ax, pts[a], pts[b], GREEN, bend=0.06, shrink=13)
    node(ax, (1.16, 0.48), "bad", RED, radius=0.08)
    arrow(ax, (0.75, 0.43), (1.07, 0.48), RED, shrink=12)
    ax.text(1.02, 0.66, "must be impossible", color=RED, ha="center", fontsize=9)
    ax.text(0.46, 0.05, "initial state in I + transitions preserve I", ha="center", fontweight="bold")
    finish(fig, "unit01-state-machine-invariant")


def fig_recursive():
    fig, (ax,) = canvas("Recursive objects are built from constructors")
    clean(ax, (0, 1), (0, 1))
    positions = {"+": (0.5, 0.84), "*": (0.3, 0.58), "3": (0.7, 0.58), "x": (0.18, 0.30), "2": (0.42, 0.30)}
    for parent, child in [("+", "*"), ("+", "3"), ("*", "x"), ("*", "2")]:
        arrow(ax, positions[parent], positions[child], GRAY, shrink=14)
    for label, p in positions.items():
        node(ax, p, label, PURPLE if label in {"+", "*"} else BLUE, radius=0.065)
    box(ax, (0.63, 0.12), 0.31, 0.13, "base: variables, constants", PALE_BLUE, BLUE, 9)
    box(ax, (0.05, 0.08), 0.34, 0.13, "constructor: combine", "#F3E8FF", PURPLE, 9)
    ax.text(0.5, 0.96, r"expression: $x\cdot2+3$", ha="center", fontweight="bold")
    finish(fig, "unit01-recursive-structure")


def fig_cantor():
    fig, (ax,) = canvas("Cantor diagonalization: construct a sequence missing from every row")
    clean(ax, (0, 8), (0, 5.3))
    data = np.array([[0, 1, 0, 1, 1], [1, 1, 0, 0, 1], [0, 0, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    for i in range(5):
        ax.text(0.55, 4.45 - 0.72 * i, f"s{i+1}", ha="right", va="center", color=GRAY)
        for j in range(5):
            color = RED if i == j else INK
            ax.text(1.15 + 0.72 * j, 4.45 - 0.72 * i, str(data[i, j]), ha="center", va="center", color=color, fontweight="bold" if i == j else None)
            if i == j:
                ax.add_patch(Rectangle((0.90 + 0.72 * j, 4.20 - 0.72 * i), 0.5, 0.5, fill=False, edgecolor=RED, lw=1.5))
    anti = 1 - np.diag(data)
    ax.text(5.15, 4.45, "flip", color=ORANGE, fontweight="bold")
    for i, bit in enumerate(anti):
        ax.text(6.2, 4.45 - 0.72 * i, str(bit), color=BLUE, ha="center", va="center", fontweight="bold")
        arrow(ax, (1.45 + 0.72 * i, 4.45 - 0.72 * i), (5.9, 4.45 - 0.72 * i), ORANGE, lw=1.0)
    ax.text(4.0, 0.42, "new sequence differs from row i at position i", ha="center", color=BLUE, fontweight="bold")
    finish(fig, "unit01-cantor-diagonal")


def fig_proof_map():
    fig, (ax,) = canvas("Proof-method selection map")
    clean(ax, (0, 1.4), (0, 1))
    box(ax, (0.50, 0.80), 0.38, 0.12, "What is the target form?", "#F3E8FF", PURPLE)
    boxes = [
        ((0.03, 0.52), "P => Q", "direct / contrapositive"),
        ((0.38, 0.52), "impossibility", "contradiction"),
        ((0.73, 0.52), "finite split", "cases"),
        ((1.08, 0.52), "all n", "induction / WOP"),
    ]
    for (x, y), top, bottom in boxes:
        box(ax, (x, y), 0.27, 0.17, top + "\n" + bottom, PALE_BLUE, BLUE, 8.5)
        arrow(ax, (0.69, 0.80), (x + 0.135, y + 0.17), GRAY, lw=1.0)
    box(ax, (0.38, 0.18), 0.62, 0.14, "Write assumptions -> justified steps -> conclusion", PALE_GREEN, GREEN, 10)
    for x in [0.165, 0.515, 0.865, 1.215]:
        arrow(ax, (x, 0.52), (0.69, 0.32), GREEN, lw=1.0)
    finish(fig, "unit01-proof-method-map")


def fig_euclid():
    fig, (ax,) = canvas("Euclidean algorithm: the gcd survives each remainder step")
    ax.axis("off")
    rows = [["252", "105", "42", r"$252=2(105)+42$"], ["105", "42", "21", r"$105=2(42)+21$"], ["42", "21", "0", r"$42=2(21)+0$"]]
    table = ax.table(cellText=rows, colLabels=["a", "b", "r", "division"], loc="center", cellLoc="center", colWidths=[.13, .13, .13, .42])
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(PALE_BLUE if r == 0 else (PALE_GREEN if r == 3 else "#F8FAFC"))
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.text(0.5, 0.18, r"$\gcd(252,105)=\gcd(105,42)=\gcd(42,21)=21$", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit02-euclidean-algorithm")


def fig_mod_clock():
    fig, (ax,) = canvas("Congruence modulo 12: addition wraps around the clock")
    clean(ax, (-1.4, 1.4), (-1.4, 1.4))
    for k in range(12):
        ang = pi / 2 - 2 * pi * k / 12
        p = (np.cos(ang), np.sin(ang))
        node(ax, p, str(k), BLUE if k not in {9, 2} else (GREEN if k == 9 else ORANGE), radius=0.095)
    a9 = pi / 2 - 2 * pi * 9 / 12
    a2 = pi / 2 - 2 * pi * 2 / 12
    arrow(ax, (0.82 * np.cos(a9), 0.82 * np.sin(a9)), (0.82 * np.cos(a2), 0.82 * np.sin(a2)), ORANGE, "+5", bend=-0.35, lw=2.5)
    ax.text(0, -1.28, r"$9+5\equiv2\ (\mathrm{mod}\ 12)$", ha="center", fontweight="bold")
    finish(fig, "unit02-modular-clock")


def fig_rsa():
    fig, (ax,) = canvas("RSA: arithmetic pipeline and the inverse exponents")
    clean(ax, (0, 1.45), (0, 1))
    items = [
        (0.02, "choose\np=5, q=11", PALE_BLUE, BLUE),
        (0.30, "$n=55$\n$\\phi=40$", PALE_BLUE, BLUE),
        (0.58, "$e=3$\n$d=27$", "#F3E8FF", PURPLE),
        (0.86, r"$c=m^e$ mod $n$", "#FEF3C7", ORANGE),
        (1.16, r"$m=c^d$ mod $n$", PALE_GREEN, GREEN),
    ]
    for x, label, fill, edge in items:
        box(ax, (x, 0.44), 0.24, 0.20, label, fill, edge, 9)
    for x in [0.26, 0.54, 0.82, 1.10]:
        arrow(ax, (x, 0.54), (x + 0.04, 0.54), GRAY)
    ax.text(0.72, 0.78, r"$ed=81\equiv1\ (\mathrm{mod}\ 40)$", ha="center", fontweight="bold", color=PURPLE)
    ax.text(0.72, 0.20, "public: (n,e)     private: d", ha="center", color=INK)
    finish(fig, "unit02-rsa-flow")


def fig_directed_walk():
    fig, (ax,) = canvas("Directed walks respect edge orientation")
    clean(ax, (0, 1), (0, 1))
    pts = {"a": (0.13, 0.50), "b": (0.39, 0.76), "c": (0.68, 0.68), "d": (0.84, 0.34), "e": (0.43, 0.25)}
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "b"), ("a", "e"), ("c", "a")]
    walk = {("a", "b"), ("b", "c"), ("c", "d")}
    for u, v in edges:
        arrow(ax, pts[u], pts[v], ORANGE if (u, v) in walk else GRAY, bend=0.04,
              lw=2.4 if (u, v) in walk else 1.2, shrink=14)
    for label, p in pts.items():
        node(ax, p, label, BLUE)
    ax.text(0.5, 0.05, r"highlighted walk: $a\to b\to c\to d$", ha="center", color=ORANGE, fontweight="bold")
    finish(fig, "unit02-directed-walk")


def fig_dag():
    fig, (ax,) = canvas("A topological order sends every DAG edge forward")
    clean(ax, (0, 1.15), (0, 1))
    pts = {"A": (0.10, 0.68), "B": (0.10, 0.30), "C": (0.40, 0.75), "D": (0.40, 0.30), "E": (0.72, 0.62), "F": (1.02, 0.46)}
    edges = [("A", "C"), ("A", "D"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F"), ("E", "F")]
    for u, v in edges:
        arrow(ax, pts[u], pts[v], GRAY, shrink=14)
    for label, p in pts.items():
        node(ax, p, label, BLUE)
    ax.text(0.1, 0.92, "1", ha="center", color=GREEN, fontweight="bold")
    ax.text(0.4, 0.92, "2", ha="center", color=GREEN, fontweight="bold")
    ax.text(0.72, 0.92, "3", ha="center", color=GREEN, fontweight="bold")
    ax.text(1.02, 0.92, "4", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit02-dag-topological-order")


def fig_hasse():
    fig, (ax,) = canvas("Hasse diagram for divisibility on the divisors of 12")
    clean(ax, (0, 1), (0, 1))
    pts = {"1": (0.5, 0.12), "2": (0.28, 0.38), "3": (0.72, 0.38), "4": (0.18, 0.66), "6": (0.55, 0.66), "12": (0.38, 0.90)}
    covers = [("1", "2"), ("1", "3"), ("2", "4"), ("2", "6"), ("3", "6"), ("4", "12"), ("6", "12")]
    for u, v in covers:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=GRAY, lw=1.8)
    for label, p in pts.items():
        node(ax, p, label, PURPLE)
    ax.text(0.86, 0.16, r"$a\preceq b:\ a|b$", ha="center", fontsize=10)
    finish(fig, "unit02-partial-order-hasse")


def fig_handshake():
    fig, (ax,) = canvas("Handshake lemma: every edge contributes two degree units")
    clean(ax, (0, 1), (0, 1))
    pts = {"A": (0.16, 0.54), "B": (0.39, 0.82), "C": (0.72, 0.74), "D": (0.83, 0.33), "E": (0.43, 0.22)}
    edges = [("A", "B"), ("A", "E"), ("B", "C"), ("B", "E"), ("C", "D"), ("C", "E"), ("D", "E")]
    deg = {v: 0 for v in pts}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=GRAY, lw=2)
    for label, p in pts.items():
        node(ax, p, label, BLUE)
        ax.text(p[0], p[1] - 0.105, f"deg={deg[label]}", ha="center", fontsize=8)
    ax.text(0.5, 0.04, r"$\sum_v\deg(v)=14=2|E|$", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit02-handshake-lemma")


def fig_coloring():
    fig, (ax,) = canvas("Proper coloring: adjacent vertices receive different colors")
    clean(ax, (-1.25, 1.25), (-1.25, 1.25))
    pts = [(np.cos(pi / 2 + 2 * pi * i / 5), np.sin(pi / 2 + 2 * pi * i / 5)) for i in range(5)]
    for i in range(5):
        a, b = pts[i], pts[(i + 1) % 5]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=2)
    colors = [BLUE, ORANGE, BLUE, ORANGE, GREEN]
    for i, (p, c) in enumerate(zip(pts, colors)):
        node(ax, p, str(i + 1), c, radius=0.12)
    ax.text(0, -1.16, r"odd cycle $C_5$: two colors fail, three colors suffice", ha="center", fontweight="bold")
    finish(fig, "unit02-graph-coloring")


def fig_spanning_tree():
    fig, (ax,) = canvas("A spanning tree keeps every vertex and removes all cycles")
    clean(ax, (0, 1), (0, 1))
    pts = {"A": (0.12, 0.58), "B": (0.35, 0.83), "C": (0.68, 0.78), "D": (0.88, 0.48), "E": (0.62, 0.20), "F": (0.28, 0.24)}
    all_edges = [("A", "B"), ("A", "F"), ("B", "C"), ("B", "F"), ("C", "D"), ("C", "E"), ("D", "E"), ("E", "F"), ("B", "E")]
    tree = {("A", "B"), ("A", "F"), ("B", "C"), ("C", "D"), ("C", "E")}
    for u, v in all_edges:
        color = GREEN if (u, v) in tree else LIGHT
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=color, lw=3 if (u, v) in tree else 1.5, zorder=1)
    for label, p in pts.items():
        node(ax, p, label, BLUE)
    ax.text(0.52, 0.04, r"$|V|=6,quad |E_{tree}|=5$", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit02-spanning-tree")


def fig_matching():
    fig, (ax,) = canvas("Stable matching: proposals move only down each preference list")
    clean(ax, (0, 1), (0, 1))
    left = {"A": (0.17, 0.76), "B": (0.17, 0.50), "C": (0.17, 0.24)}
    right = {"X": (0.83, 0.76), "Y": (0.83, 0.50), "Z": (0.83, 0.24)}
    matching = [("A", "Y"), ("B", "X"), ("C", "Z")]
    rejected = [("A", "X"), ("C", "Y")]
    for u, v in rejected:
        ax.plot([left[u][0], right[v][0]], [left[u][1], right[v][1]], color=PALE_RED, lw=2, ls="--")
    for u, v in matching:
        ax.plot([left[u][0], right[v][0]], [left[u][1], right[v][1]], color=GREEN, lw=3)
    for label, p in {**left, **right}.items():
        node(ax, p, label, BLUE if label in left else PURPLE, radius=0.065)
    ax.text(0.5, 0.92, "solid = final matching; dashed = rejected proposal", ha="center", fontsize=9)
    ax.text(0.5, 0.08, "stability excludes a mutually preferred blocking pair", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit02-stable-matching")


def fig_sum_product():
    fig, (ax,) = canvas("Sum rule versus product rule")
    clean(ax, (0, 1), (0, 1))
    box(ax, (0.38, 0.79), 0.24, 0.12, "count outcomes", "#F3E8FF", PURPLE)
    box(ax, (0.06, 0.50), 0.32, 0.15, "disjoint alternatives\nSUM", PALE_BLUE, BLUE)
    box(ax, (0.62, 0.50), 0.32, 0.15, "successive choices\nPRODUCT", PALE_GREEN, GREEN)
    arrow(ax, (0.46, 0.79), (0.25, 0.65), BLUE, "or")
    arrow(ax, (0.54, 0.79), (0.75, 0.65), GREEN, "then")
    ax.text(0.22, 0.29, r"$|A\cup B|=|A|+|B|$", ha="center", fontweight="bold", color=BLUE)
    ax.text(0.78, 0.29, r"$|A\times B|=|A||B|$", ha="center", fontweight="bold", color=GREEN)
    ax.text(0.5, 0.10, "First ask whether the cases overlap and whether choices are independent stages.", ha="center", fontsize=9)
    finish(fig, "unit03-sum-product-rule")


def fig_binomial_paths():
    fig, (ax,) = canvas("Binomial coefficients count lattice paths")
    clean(ax, (-0.5, 5.7), (-0.5, 3.8))
    for x in range(6):
        for y in range(4):
            ax.scatter(x, y, s=24, color=GRAY, zorder=2)
            if x < 5:
                ax.plot([x, x + 1], [y, y], color=LIGHT, lw=1)
            if y < 3:
                ax.plot([x, x], [y, y + 1], color=LIGHT, lw=1)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (5, 3)]
    ax.plot([p[0] for p in path], [p[1] for p in path], color=ORANGE, lw=3, drawstyle="steps-post")
    node(ax, (0, 0), "S", GREEN, radius=0.12)
    node(ax, (5, 3), "T", BLUE, radius=0.12)
    ax.text(2.5, 3.45, r"5 rights + 3 ups: $\binom{8}{3}=56$ paths", ha="center", fontweight="bold")
    finish(fig, "unit03-binomial-paths")


def fig_stars_bars():
    fig, (ax,) = canvas("Stars and bars encode nonnegative integer solutions")
    clean(ax, (0, 10), (0, 3.5))
    symbols = ["*", "*", "|", "*", "*", "*", "|", "*", "*"]
    colors = [BLUE, BLUE, ORANGE, GREEN, GREEN, GREEN, ORANGE, PURPLE, PURPLE]
    for i, (s, c) in enumerate(zip(symbols, colors)):
        ax.text(i + 0.8, 2.25, s, fontsize=33, ha="center", va="center", color=c, fontweight="bold")
    ax.text(1.3, 1.25, r"$x_1=2$", ha="center", color=BLUE, fontweight="bold")
    ax.text(4.3, 1.25, r"$x_2=3$", ha="center", color=GREEN, fontweight="bold")
    ax.text(7.7, 1.25, r"$x_3=2$", ha="center", color=PURPLE, fontweight="bold")
    ax.text(5, 0.40, r"$x_1+x_2+x_3=7$ has $\binom{9}{2}=36$ solutions", ha="center", fontweight="bold")
    finish(fig, "unit03-stars-and-bars")


def fig_pigeonhole():
    fig, (ax,) = canvas("Pigeonhole principle: more objects than boxes forces a collision")
    clean(ax, (0, 1), (0, 1))
    box_x = [0.08, 0.38, 0.68]
    for i, x in enumerate(box_x):
        ax.add_patch(Rectangle((x, 0.15), 0.22, 0.30, facecolor="#F8FAFC", edgecolor=INK, lw=1.8))
        ax.text(x + 0.11, 0.09, f"box {i+1}", ha="center", fontsize=9)
    balls = [(0.14, 0.30, BLUE), (0.24, 0.30, ORANGE), (0.49, 0.30, GREEN), (0.79, 0.30, PURPLE)]
    for x, y, c in balls:
        ax.add_patch(Circle((x, y), 0.045, facecolor=c, edgecolor="white"))
    ax.text(0.5, 0.73, r"$N=4$ objects, $k=3$ boxes", ha="center", fontweight="bold")
    ax.text(0.5, 0.60, r"some box contains at least $\lceil N/k\rceil=2$ objects", ha="center", color=RED)
    finish(fig, "unit03-pigeonhole-principle")


def fig_inclusion_exclusion():
    fig, (ax,) = canvas("Inclusion-exclusion corrects double counting")
    clean(ax, (0, 1), (0, 1))
    ax.add_patch(Circle((0.42, 0.55), 0.29, facecolor=PALE_BLUE, edgecolor=BLUE, lw=2, alpha=0.85))
    ax.add_patch(Circle((0.64, 0.55), 0.29, facecolor=PALE_GREEN, edgecolor=GREEN, lw=2, alpha=0.85))
    ax.text(0.28, 0.78, "A: 18", color=BLUE, fontweight="bold")
    ax.text(0.69, 0.78, "B: 15", color=GREEN, fontweight="bold")
    ax.text(0.53, 0.55, "7", ha="center", va="center", color=RED, fontweight="bold", fontsize=15)
    ax.text(0.5, 0.14, r"$|A\cup B|=18+15-7=26$", ha="center", fontweight="bold")
    ax.text(0.5, 0.07, "subtract the overlap exactly once", ha="center", color=RED, fontsize=9)
    finish(fig, "unit03-inclusion-exclusion")


def fig_sample_space():
    fig, (ax,) = canvas("Events are subsets of a sample space")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("second die")
    ax.set_ylabel("first die")
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.grid(color=LIGHT, lw=1)
    for i in range(1, 7):
        for j in range(1, 7):
            color = GREEN if i + j == 7 else (PALE_BLUE if (i + j) % 2 == 0 else "white")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="none", zorder=0))
    ax.text(3.5, 6.65, "green: sum = 7; blue: even sum", ha="center", fontsize=9)
    finish(fig, "unit04-sample-space-events")


def fig_bayes():
    fig, (ax,) = canvas("Bayes tree: posterior probability follows the surviving mass")
    clean(ax, (0, 1), (0, 1))
    root = (0.08, 0.50)
    d, nd = (0.40, 0.75), (0.40, 0.28)
    dp, dn, ndp, ndn = (0.78, 0.86), (0.78, 0.64), (0.78, 0.38), (0.78, 0.16)
    node(ax, root, "start", PURPLE, radius=0.07)
    for p, label, color in [(d, "D", RED), (nd, "not D", BLUE), (dp, "+", RED), (dn, "-", GREEN), (ndp, "+", ORANGE), (ndn, "-", GREEN)]:
        node(ax, p, label, color, radius=0.065)
    for a, b, t in [(root, d, ".01"), (root, nd, ".99"), (d, dp, ".90"), (d, dn, ".10"), (nd, ndp, ".05"), (nd, ndn, ".95")]:
        arrow(ax, a, b, GRAY, t, lw=1.2, shrink=17)
    ax.text(0.52, 0.03, r"$P(D\mid +)=.009/(.009+.0495)\approx.154$", ha="center", fontweight="bold")
    finish(fig, "unit04-bayes-tree")


def fig_independence():
    fig, (ax,) = canvas("Independent events multiply their probabilities")
    clean(ax, (0, 1), (0, 1))
    ax.add_patch(Rectangle((0.12, 0.13), 0.76, 0.72, facecolor="#F8FAFC", edgecolor=INK, lw=1.5))
    ax.add_patch(Rectangle((0.12, 0.13), 0.38, 0.72, facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.5, alpha=0.8))
    ax.add_patch(Rectangle((0.12, 0.13), 0.76, 0.24, facecolor=PALE_GREEN, edgecolor=GREEN, lw=1.5, alpha=0.75))
    ax.add_patch(Rectangle((0.12, 0.13), 0.38, 0.24, facecolor=ORANGE, edgecolor="none", alpha=0.65))
    ax.text(0.31, 0.88, r"$P(A)=1/2$", ha="center", color=BLUE, fontweight="bold")
    ax.text(0.92, 0.25, r"$P(B)=1/3$", va="center", color=GREEN, fontweight="bold", rotation=90)
    ax.text(0.31, 0.25, r"$P(A\cap B)=1/6$", ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.05, r"$P(A\cap B)=P(A)P(B)$", ha="center", fontweight="bold")
    finish(fig, "unit04-independence-grid")


def fig_pmf():
    fig, (ax,) = canvas("A probability mass function assigns mass to each value")
    x = np.arange(5)
    p = np.array([comb(4, k) / 16 for k in x])
    ax.bar(x, p, color=BLUE, edgecolor="white", width=0.72)
    ax.set_xlabel("x = number of heads in 4 fair tosses")
    ax.set_ylabel(r"$P(X=x)$")
    ax.set_xticks(x)
    ax.set_ylim(0, 0.45)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, pi_ in zip(x, p):
        ax.text(xi, pi_ + 0.015, f"{pi_:.3f}", ha="center", fontsize=9)
    ax.text(2, 0.41, r"$\sum_x p_X(x)=1$", ha="center", color=GREEN, fontweight="bold")
    finish(fig, "unit04-random-variable-pmf")


def fig_expectation_variance():
    fig, axes = canvas("Expectation locates the center; variance measures spread", 2)
    distributions = [
        (np.array([1, 2, 3]), np.array([0.25, 0.5, 0.25]), BLUE, "small variance"),
        (np.array([0, 2, 4]), np.array([0.25, 0.5, 0.25]), ORANGE, "large variance"),
    ]
    for ax, (x, p, color, label) in zip(axes, distributions):
        ax.bar(x, p, color=color, width=0.55)
        ax.axvline(2, color=GREEN, ls="--", lw=2)
        mean = float(np.dot(x, p))
        var = float(np.dot((x - mean) ** 2, p))
        ax.set_title(f"{label}\nmean={mean:.0f}, variance={var:.1f}", fontsize=11)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 0.62)
        ax.set_xticks(range(5))
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "unit04-expectation-variance")


def fig_concentration():
    fig, (ax,) = canvas("Concentration inequalities bound tail probability")
    x = np.arange(0, 21)
    p = np.array([comb(20, k) for k in x], dtype=float) / 2**20
    ax.bar(x, p, color=BLUE, width=0.82, edgecolor="white")
    tail = np.abs(x - 10) >= 5
    ax.bar(x[tail], p[tail], color=RED, width=0.82, edgecolor="white")
    actual = p[tail].sum()
    cheb = 5 / 25
    ax.axvline(10, color=GREEN, ls="--", lw=2)
    ax.set_xlabel(r"$X\sim Binomial(20,1/2)$")
    ax.set_ylabel("probability")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(10, 0.19, r"$E[X]=10, Var(X)=5$", ha="center", color=GREEN, fontweight="bold")
    ax.text(10, 0.165, f"actual tail = {actual:.3f}; Chebyshev <= {cheb:.3f}", ha="center", fontsize=9)
    finish(fig, "unit04-concentration-bounds")


def fig_pagerank():
    fig, axes = canvas("Random walk and PageRank: stationary flow balances each node", 2)
    ax, bx = axes
    clean(ax, (0, 1), (0, 1))
    pts = {"A": (0.18, 0.55), "B": (0.52, 0.78), "C": (0.80, 0.46), "D": (0.48, 0.22)}
    edges = [("A", "B", "1/2"), ("A", "D", "1/2"), ("B", "C", "1"), ("C", "A", "1/2"), ("C", "D", "1/2"), ("D", "C", "1")]
    for u, v, t in edges:
        arrow(ax, pts[u], pts[v], ORANGE, t, bend=0.05, lw=1.5, shrink=16)
    for label, p in pts.items():
        node(ax, p, label, BLUE)
    P = np.array([[0, .5, 0, .5], [0, 0, 1, 0], [.5, 0, 0, .5], [0, 0, 1, 0]])
    vals, vecs = np.linalg.eig(P.T)
    stationary = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    stationary /= stationary.sum()
    bx.bar(list("ABCD"), stationary, color=[BLUE, ORANGE, GREEN, PURPLE])
    bx.set_ylim(0, 0.55)
    bx.set_title("stationary distribution", fontsize=11)
    bx.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(stationary):
        bx.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    finish(fig, "unit04-random-walk-pagerank")


def mathematical_self_checks() -> None:
    implication = [((p and (not q)) is False) for p, q in [(True, True), (True, False), (False, True), (False, False)]]
    assert implication == [True, False, True, True]
    assert gcd(252, 105) == 21
    assert (3 * 27) % 40 == 1
    assert all(pow(pow(m, 3, 55), 27, 55) == m for m in range(55))
    assert sum([2, 3, 3, 2, 4]) == 2 * 7
    assert comb(8, 3) == 56
    assert comb(9, 2) == 36
    assert 18 + 15 - 7 == 26
    assert abs((.01 * .9) / (.01 * .9 + .99 * .05) - 0.15384615384615385) < 1e-12
    pmf = np.array([comb(4, k) / 16 for k in range(5)])
    assert abs(pmf.sum() - 1) < 1e-12
    assert abs(np.dot(np.arange(5), pmf) - 2) < 1e-12
    P = np.array([[0, .5, 0, .5], [0, 0, 1, 0], [.5, 0, 0, .5], [0, 0, 1, 0]])
    assert np.allclose(P.sum(axis=1), 1)


FIGURES = [
    fig_implication, fig_sets, fig_induction, fig_wop, fig_invariant, fig_recursive,
    fig_cantor, fig_proof_map, fig_euclid, fig_mod_clock, fig_rsa, fig_directed_walk,
    fig_dag, fig_hasse, fig_handshake, fig_coloring, fig_spanning_tree, fig_matching,
    fig_sum_product, fig_binomial_paths, fig_stars_bars, fig_pigeonhole,
    fig_inclusion_exclusion, fig_sample_space, fig_bayes, fig_independence, fig_pmf,
    fig_expectation_variance, fig_concentration, fig_pagerank,
]


if __name__ == "__main__":
    mathematical_self_checks()
    for make_figure in FIGURES:
        make_figure()
    outputs = sorted(OUT.glob("unit*.png"))
    assert len(outputs) == 30, f"expected 30 PNG files, found {len(outputs)}"
    assert all(plt.imread(path).shape[:2] == (900, 1600) for path in outputs)
    print(f"Generated {len(outputs)} checked figures in {OUT}")
