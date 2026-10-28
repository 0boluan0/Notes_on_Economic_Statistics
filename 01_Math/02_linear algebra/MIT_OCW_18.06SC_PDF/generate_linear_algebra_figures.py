#!/usr/bin/env python3
"""Generate deterministic PNG figures for the MIT 18.06SC course notes."""

from pathlib import Path
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("Install numpy and matplotlib before running this script.") from exc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "98_attachment" / "linear_algebra" / "mit18_06sc"

BLUE = "#2563A6"
ORANGE = "#D97706"
GREEN = "#2E8B57"
RED = "#C43D3D"
PURPLE = "#7551A8"
DARK = "#293241"
MID = "#66788A"
LIGHT = "#D9E2EC"
PALE_BLUE = "#DCEBFA"
PALE_GREEN = "#DDF2E5"
PALE_ORANGE = "#FCE8C3"


def close_enough(a, b, tolerance=1e-9):
    return np.allclose(a, b, rtol=tolerance, atol=tolerance)


def new_figure(columns=1):
    fig, axes = plt.subplots(1, columns, figsize=(8, 4.5), dpi=200)
    return fig, np.atleast_1d(axes)


def style_axis(ax, x_label=r"$x_1$", y_label=r"$x_2$", grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(DARK)
    ax.tick_params(colors=DARK, labelsize=8)
    ax.set_xlabel(x_label, color=DARK, fontsize=10)
    ax.set_ylabel(y_label, color=DARK, fontsize=10, rotation=0, labelpad=10)
    if grid:
        ax.grid(True, color=LIGHT, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def arrow(ax, start, end, color=DARK, width=1.7, mutation=12, style="->"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def vector(ax, value, color, label=None, origin=(0, 0), width=2.0):
    value = np.asarray(value, dtype=float)
    origin = np.asarray(origin, dtype=float)
    arrow(ax, origin, origin + value, color=color, width=width)
    if label:
        point = origin + value
        ax.text(point[0] + 0.08, point[1] + 0.08, label, color=color, fontsize=9)


def matrix_panel(ax, matrix, title, cmap="Blues", fmt="{:.0f}"):
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap=cmap, vmin=np.min(matrix), vmax=np.max(matrix), alpha=0.78)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", color=DARK, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=DARK, fontsize=11)
    for spine in ax.spines.values():
        spine.set_color(DARK)
        spine.set_linewidth(1.1)


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)


def unit1_row_column_picture():
    fig, (ax, bx) = new_figure(2)
    x = np.linspace(-0.6, 2.4, 300)
    y1 = 2 * x
    y2 = (x + 3) / 2
    solution = np.linalg.solve(np.array([[2.0, -1.0], [-1.0, 2.0]]), np.array([0.0, 3.0]))
    assert close_enough(solution, [1, 2])
    ax.plot(x, y1, color=BLUE, linewidth=2.2, label=r"$2x-y=0$")
    ax.plot(x, y2, color=ORANGE, linewidth=2.2, label=r"$-x+2y=3$")
    ax.scatter(*solution, color=GREEN, s=40, zorder=5)
    ax.annotate(r"$x=(1,2)$", solution, xytext=(12, -25), textcoords="offset points", color=GREEN)
    ax.set_title("Row picture: intersection", fontsize=11, color=DARK)
    style_axis(ax, r"$x$", r"$y$")
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(-0.5, 4.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    a1 = np.array([2.0, -1.0])
    a2 = np.array([-1.0, 2.0])
    b = a1 + 2 * a2
    assert close_enough(b, [0, 3])
    vector(bx, a1, BLUE, r"$a_1$")
    vector(bx, 2 * a2, ORANGE, r"$2a_2$")
    vector(bx, b, GREEN, r"$b=a_1+2a_2$")
    bx.plot([a1[0], b[0]], [a1[1], b[1]], color=ORANGE, linestyle="--", linewidth=1.2)
    bx.plot([2 * a2[0], b[0]], [2 * a2[1], b[1]], color=BLUE, linestyle="--", linewidth=1.2)
    bx.set_title("Column picture: combination", fontsize=11, color=DARK)
    style_axis(bx)
    bx.set_aspect("equal")
    bx.set_xlim(-2.5, 2.8)
    bx.set_ylim(-1.6, 4.4)
    fig.suptitle(r"The same equation $Ax=b$ in two geometries", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "mit18.06sc-unit1-row-column-picture")


def unit1_elimination_lu():
    A = np.array([[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]])
    L = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, -1.0, 1.0]])
    U = np.array([[2.0, 1.0, 1.0], [0.0, -8.0, -2.0], [0.0, 0.0, 1.0]])
    assert close_enough(L @ U, A)
    fig, axes = new_figure(3)
    matrix_panel(axes[0], A, r"$A$")
    matrix_panel(axes[1], L, r"$L$ stores multipliers", cmap="Greens")
    matrix_panel(axes[2], U, r"$U$ after elimination", cmap="Oranges")
    fig.text(0.325, 0.47, r"$=$", fontsize=22, color=DARK, ha="center")
    fig.text(0.665, 0.47, r"$\times$", fontsize=22, color=DARK, ha="center")
    fig.suptitle(r"Elimination organized as $A=LU$", fontsize=13, color=DARK)
    fig.text(0.5, 0.06, r"$m_{21}=2,\quad m_{31}=-1,\quad m_{32}=-1$", ha="center", color=MID, fontsize=10)
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.90), w_pad=2.5)
    save(fig, "mit18.06sc-unit1-elimination-lu")


def unit1_affine_solution():
    fig, (ax,) = new_figure()
    xp = np.array([3.0, 0.0])
    n = np.array([-2.0, 1.0])
    assert close_enough(np.array([[1.0, 2.0]]) @ xp, [3.0])
    assert close_enough(np.array([[1.0, 2.0]]) @ n, [0.0])
    t = np.linspace(-1.7, 1.6, 300)
    points = xp[:, None] + n[:, None] * t
    ax.plot(points[0], points[1], color=BLUE, linewidth=2.6, label=r"$x_p+N(A)$")
    ax.scatter(*xp, color=ORANGE, s=45, zorder=5)
    vector(ax, 0.75 * n, GREEN, r"$n\in N(A)$", origin=xp)
    ax.text(xp[0] + 0.12, xp[1] - 0.35, r"$x_p=(3,0)$", color=ORANGE, fontsize=9)
    ax.text(-0.7, 2.25, r"$A=[1\;2],\quad Ax=3$", color=DARK, fontsize=11)
    ax.text(-0.7, 1.85, r"$x=x_p+tn=(3,0)+t(-2,1)$", color=DARK, fontsize=10)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 6.2)
    ax.set_ylim(-2.0, 3.1)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("A consistent underdetermined system has an affine solution set", color=DARK, fontsize=12)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit1-affine-solution")


def unit1_four_subspaces():
    fig, (ax,) = new_figure()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = {
        "row": (0.7, 3.5, 3.1, 1.25, PALE_BLUE, r"$C(A^T)\subset\mathbb{R}^n$", r"dimension $r$"),
        "null": (0.7, 0.9, 3.1, 1.25, PALE_ORANGE, r"$N(A)\subset\mathbb{R}^n$", r"dimension $n-r$"),
        "col": (6.2, 3.5, 3.1, 1.25, PALE_GREEN, r"$C(A)\subset\mathbb{R}^m$", r"dimension $r$"),
        "left": (6.2, 0.9, 3.1, 1.25, "#F2E7FA", r"$N(A^T)\subset\mathbb{R}^m$", r"dimension $m-r$"),
    }
    for x, y, w, h, face, title, subtitle in boxes.values():
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=DARK, linewidth=1.2))
        ax.text(x + w / 2, y + 0.74, title, ha="center", va="center", color=DARK, fontsize=11)
        ax.text(x + w / 2, y + 0.32, subtitle, ha="center", va="center", color=MID, fontsize=9)
    arrow(ax, (3.8, 4.12), (6.2, 4.12), BLUE, 2.2)
    ax.text(5.0, 4.42, r"$A$: one-to-one onto", ha="center", color=BLUE, fontsize=10)
    ax.add_patch(Rectangle((4.8, 1.17), 0.9, 0.7, facecolor="white", edgecolor=ORANGE, linewidth=1.3))
    ax.text(5.25, 1.52, r"$\{0\}$", ha="center", va="center", color=ORANGE, fontsize=10)
    arrow(ax, (3.8, 1.52), (4.8, 1.52), ORANGE, 2.0)
    ax.text(4.30, 1.78, r"$A$", ha="center", color=ORANGE, fontsize=10)
    ax.plot([2.25, 2.25], [2.15, 3.5], color=RED, linewidth=1.4, linestyle="--")
    ax.plot([7.75, 7.75], [2.15, 3.5], color=RED, linewidth=1.4, linestyle="--")
    ax.text(2.45, 2.72, r"$\perp$", color=RED, fontsize=16)
    ax.text(7.95, 2.72, r"$\perp$", color=RED, fontsize=16)
    ax.text(2.25, 5.35, r"Input space: $\mathbb{R}^n=C(A^T)\oplus N(A)$", ha="center", color=DARK, fontsize=10)
    ax.text(7.75, 5.35, r"Output space: $\mathbb{R}^m=C(A)\oplus N(A^T)$", ha="center", color=DARK, fontsize=10)
    ax.set_title("The four fundamental subspaces and the action of A", color=DARK, fontsize=13)
    save(fig, "mit18.06sc-unit1-four-subspaces")


def unit1_rank_one():
    u = np.array([2.0, 1.0, -1.0])
    v = np.array([1.0, -2.0])
    A = np.outer(u, v)
    assert np.linalg.matrix_rank(A) == 1
    fig, (ax, bx) = new_figure(2)
    matrix_panel(ax, A, r"$A=uv^T$", cmap="PuBu", fmt="{:.0f}")
    bx.axline((0, 0), slope=u[1] / u[0], color=BLUE, linewidth=2.5, label=r"$C(A)=\mathrm{span}(u)$")
    samples = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, 0.5])]
    for idx, x in enumerate(samples):
        y3 = A @ x
        y = y3[:2]
        assert abs(np.linalg.det(np.column_stack((u[:2], y)))) < 1e-9
        vector(bx, y, [ORANGE, GREEN, PURPLE, RED][idx], rf"$Ax_{idx+1}$", width=1.5)
    bx.text(-3.7, 3.2, r"$Ax=u(v^Tx)$", fontsize=11, color=DARK)
    bx.text(-3.7, 2.75, "every output lies on one line", fontsize=9, color=MID)
    style_axis(bx, r"$y_1$", r"$y_2$")
    bx.set_aspect("equal")
    bx.set_xlim(-4, 4)
    bx.set_ylim(-2.6, 3.8)
    bx.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Rank-one matrices collapse the input to one output direction", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "mit18.06sc-unit1-rank-one")


def unit1_incidence_network():
    nodes = np.array([[0.5, 1.9], [2.5, 3.2], [4.6, 1.9], [2.5, 0.35]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    incidence = np.zeros((len(edges), len(nodes)))
    for i, (tail, head) in enumerate(edges):
        incidence[i, tail] = -1
        incidence[i, head] = 1
    assert close_enough(incidence @ np.ones(4), np.zeros(len(edges)))
    assert np.linalg.matrix_rank(incidence) == 3
    fig, (ax, bx) = new_figure(2)
    for i, (tail, head) in enumerate(edges):
        start = nodes[tail]
        end = nodes[head]
        direction = end - start
        s = start + 0.18 * direction / np.linalg.norm(direction)
        e = end - 0.22 * direction / np.linalg.norm(direction)
        arrow(ax, s, e, ORANGE if i == 4 else BLUE, 1.8)
        mid = (start + end) / 2
        ax.text(mid[0], mid[1] + 0.12, rf"$e_{i+1}$", color=DARK, fontsize=8)
    for i, point in enumerate(nodes):
        ax.add_patch(Circle(point, 0.18, facecolor=PALE_GREEN, edgecolor=DARK, linewidth=1.3, zorder=4))
        ax.text(*point, rf"$v_{i+1}$", ha="center", va="center", fontsize=8, color=DARK, zorder=5)
    ax.set_xlim(0, 5.1)
    ax.set_ylim(-0.1, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Oriented network", color=DARK, fontsize=11)
    matrix_panel(bx, incidence, r"Incidence matrix $A$ (edges x nodes)", cmap="RdBu", fmt="{:.0f}")
    fig.text(0.72, 0.08, r"$A\mathbf{1}=0,\quad \mathrm{rank}(A)=|V|-1$ for a connected graph", ha="center", color=DARK, fontsize=10)
    fig.suptitle("Differences on edges come from node potentials", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    save(fig, "mit18.06sc-unit1-incidence-network")


def unit2_orthogonal_complements():
    fig, (ax,) = new_figure()
    d = np.array([2.0, 1.0])
    p = np.array([-1.0, 2.0])
    assert close_enough(d @ p, 0)
    t = np.linspace(-2.4, 2.4, 100)
    ax.plot(t * d[0], t * d[1], color=BLUE, linewidth=2.5, label=r"$S=C(A)$")
    ax.plot(t * p[0], t * p[1], color=ORANGE, linewidth=2.5, label=r"$S^\perp=N(A^T)$")
    vector(ax, 1.2 * d, BLUE, r"$s$")
    vector(ax, 0.95 * p, ORANGE, r"$e$")
    ax.add_patch(Rectangle((0, 0), 0.28, 0.28, angle=np.degrees(np.arctan2(d[1], d[0])), fill=False, edgecolor=RED, linewidth=1.2))
    ax.text(-4.3, 4.0, r"$s^Te=0$", color=DARK, fontsize=12)
    style_axis(ax, r"$y_1$", r"$y_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4.5, 4.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("A subspace and its orthogonal complement", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-orthogonal-complements")


def unit2_projection():
    a = np.array([3.0, 1.5])
    b = np.array([2.0, 4.0])
    p = a * (a @ b) / (a @ a)
    e = b - p
    assert close_enough(a @ e, 0)
    fig, (ax,) = new_figure()
    t = np.linspace(-0.8, 1.8, 100)
    ax.plot(t * a[0], t * a[1], color=BLUE, linewidth=2.5, label=r"$\mathrm{span}(a)$")
    vector(ax, b, ORANGE, r"$b$")
    vector(ax, p, GREEN, r"$p=Pb$")
    arrow(ax, p, b, RED, 2.0)
    ax.text((p[0] + b[0]) / 2 + 0.12, (p[1] + b[1]) / 2, r"$e=b-p$", color=RED, fontsize=9)
    ax.scatter(*p, color=GREEN, s=38, zorder=5)
    ax.text(-1.8, 4.5, r"$p=\dfrac{a^Tb}{a^Ta}a,\qquad a^T(b-p)=0$", fontsize=11, color=DARK)
    style_axis(ax, r"$y_1$", r"$y_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 5.8)
    ax.set_ylim(-1.2, 5.1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Orthogonal projection decomposes b into p + e", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-projection")


def unit2_least_squares_fit():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 2.0, 4.0])
    A = np.column_stack((np.ones_like(t), t))
    coeff = np.linalg.solve(A.T @ A, A.T @ y)
    residual = y - A @ coeff
    assert close_enough(A.T @ residual, [0, 0])
    grid = np.linspace(-0.25, 3.35, 200)
    line = coeff[0] + coeff[1] * grid
    fig, (ax,) = new_figure()
    ax.scatter(t, y, color=BLUE, s=45, label="data", zorder=4)
    ax.plot(grid, line, color=ORANGE, linewidth=2.4, label=r"$\hat y=c+dt$")
    fit = A @ coeff
    for xi, yi, pi in zip(t, y, fit):
        ax.plot([xi, xi], [pi, yi], color=RED, linewidth=1.5)
    ax.text(-0.08, 4.25, rf"$\hat y={coeff[0]:.2f}+{coeff[1]:.2f}t$", color=DARK, fontsize=11)
    ax.text(-0.08, 3.88, r"$A^T(b-A\hat x)=0$", color=RED, fontsize=10)
    style_axis(ax, r"$t$", r"$y$")
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(0.3, 4.7)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Least squares chooses a line with orthogonal residual", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-least-squares-fit")


def unit2_gram_schmidt():
    a1 = np.array([3.0, 1.0])
    a2 = np.array([1.0, 3.0])
    proj = a1 * (a1 @ a2) / (a1 @ a1)
    w = a2 - proj
    q1 = a1 / np.linalg.norm(a1)
    q2 = w / np.linalg.norm(w)
    assert close_enough(q1 @ q2, 0)
    assert close_enough(np.column_stack((q1, q2)).T @ np.column_stack((q1, q2)), np.eye(2))
    fig, (ax,) = new_figure()
    vector(ax, a1, BLUE, r"$a_1$")
    vector(ax, a2, ORANGE, r"$a_2$")
    vector(ax, proj, PURPLE, r"$\mathrm{proj}_{a_1}a_2$")
    arrow(ax, proj, a2, GREEN, 2.1)
    ax.text((proj[0] + a2[0]) / 2 - 0.55, (proj[1] + a2[1]) / 2 + 0.05, r"$w_2$", color=GREEN, fontsize=10)
    vector(ax, 1.7 * q1, BLUE, r"$q_1$", origin=(-2.1, -1.0), width=1.4)
    vector(ax, 1.7 * q2, GREEN, r"$q_2$", origin=(-2.1, -1.0), width=1.4)
    ax.text(-2.7, 3.9, r"$w_2=a_2-\mathrm{proj}_{a_1}a_2$", fontsize=11, color=DARK)
    ax.text(-2.7, 3.5, r"$Q^TQ=I$", fontsize=10, color=DARK)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-3.0, 4.2)
    ax.set_ylim(-1.8, 4.4)
    ax.set_title("Gram--Schmidt removes the component already explained", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-gram-schmidt")


def unit2_determinant_volume():
    a = np.array([3.0, 0.8])
    b = np.array([1.0, 2.5])
    det = np.linalg.det(np.column_stack((a, b)))
    assert det > 0 and close_enough(det, 6.7)
    vertices = np.array([[0, 0], a, a + b, b])
    fig, (ax,) = new_figure()
    ax.add_patch(Polygon(vertices, closed=True, facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=2, alpha=0.9))
    vector(ax, a, BLUE, r"$a$")
    vector(ax, b, ORANGE, r"$b$")
    ax.plot([a[0], (a + b)[0]], [a[1], (a + b)[1]], color=ORANGE, linestyle="--")
    ax.plot([b[0], (a + b)[0]], [b[1], (a + b)[1]], color=BLUE, linestyle="--")
    ax.text(1.55, 1.65, rf"area $=|\det[a\ b]|={abs(det):.1f}$", color=DARK, fontsize=11, ha="center")
    ax.text(-0.75, 3.45, r"positive sign = counterclockwise orientation", color=GREEN, fontsize=9)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 4.7)
    ax.set_ylim(-0.8, 3.9)
    ax.set_title("The determinant is signed volume scaling", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-determinant-volume")


def unit2_eigenvectors():
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-1.0, 1.0]) / np.sqrt(2)
    assert close_enough(A @ v1, 3 * v1)
    assert close_enough(A @ v2, 2 * v2)
    fig, (ax, bx) = new_figure(2)
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    image = A @ circle
    ax.plot(circle[0], circle[1], color=LIGHT, linewidth=2)
    vector(ax, v1, BLUE, r"$v_1$")
    vector(ax, v2, ORANGE, r"$v_2$")
    ax.set_title("before A", color=DARK, fontsize=11)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    bx.plot(image[0], image[1], color=PALE_BLUE, linewidth=2.2)
    vector(bx, A @ v1, BLUE, r"$Av_1=3v_1$")
    vector(bx, A @ v2, ORANGE, r"$Av_2=2v_2$")
    bx.set_title("after A", color=DARK, fontsize=11)
    style_axis(bx, r"$y_1$", r"$y_2$")
    bx.set_aspect("equal")
    bx.set_xlim(-3.7, 3.7)
    bx.set_ylim(-2.8, 2.8)
    fig.suptitle("Eigenvectors keep their direction under the transformation", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "mit18.06sc-unit2-eigenvectors")


def unit2_diagonalization():
    fig, (ax,) = new_figure()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.5, PALE_BLUE, r"$x$", "standard coordinates"),
        (3.2, "#F2E7FA", r"$c=S^{-1}x$", "eigen-coordinates"),
        (6.2, PALE_ORANGE, r"$\Lambda^k c$", "scale each mode"),
        (9.3, PALE_GREEN, r"$A^kx=S\Lambda^kS^{-1}x$", "return to space"),
    ]
    for x, face, top, bottom in boxes:
        ax.add_patch(Rectangle((x, 1.55), 2.1, 1.55, facecolor=face, edgecolor=DARK, linewidth=1.2))
        ax.text(x + 1.05, 2.55, top, ha="center", va="center", color=DARK, fontsize=11)
        ax.text(x + 1.05, 1.95, bottom, ha="center", va="center", color=MID, fontsize=8)
    for x1, x2, label in [(2.6, 3.2, r"$S^{-1}$"), (5.3, 6.2, r"$\Lambda^k$"), (8.3, 9.3, r"$S$")]:
        arrow(ax, (x1, 2.32), (x2, 2.32), BLUE, 1.8)
        ax.text((x1 + x2) / 2, 2.65, label, ha="center", color=BLUE, fontsize=9)
    ax.text(6.0, 4.25, r"$A=S\Lambda S^{-1}\quad\Longrightarrow\quad A^k=S\Lambda^kS^{-1}$", ha="center", color=DARK, fontsize=13)
    ax.text(6.0, 0.65, "Diagonalization turns repeated matrix multiplication into scalar powers.", ha="center", color=MID, fontsize=10)
    save(fig, "mit18.06sc-unit2-diagonalization")


def unit2_matrix_exponential():
    fig, (ax,) = new_figure()
    x = np.linspace(-3, 3, 24)
    y = np.linspace(-3, 3, 24)
    X, Y = np.meshgrid(x, y)
    U = 0.55 * X
    V = -1.0 * Y
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(X, Y, U, V, color=speed, cmap="Blues", density=1.2, linewidth=0.9, arrowsize=0.8)
    for c1, c2, color in [(0.25, 2.5, ORANGE), (-0.25, 2.5, GREEN), (0.45, -2.0, RED)]:
        t = np.linspace(-1.0, 2.3, 240)
        xt = c1 * np.exp(0.55 * t)
        yt = c2 * np.exp(-t)
        ax.plot(xt, yt, color=color, linewidth=2)
    ax.axhline(0, color=BLUE, linewidth=1.5, linestyle="--", label=r"unstable eigendirection $\lambda=0.55$")
    ax.axvline(0, color=ORANGE, linewidth=1.5, linestyle="--", label=r"stable eigendirection $\lambda=-1$")
    ax.text(-2.85, 2.65, r"$u'(t)=Au(t),\quad u(t)=e^{At}u(0)$", color=DARK, fontsize=11)
    style_axis(ax, r"$u_1$", r"$u_2$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Matrix exponentials separate stable and unstable modes", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-matrix-exponential")


def unit2_markov_steady_state():
    A = np.array([[0.2, 0.4, 0.3], [0.4, 0.2, 0.3], [0.4, 0.4, 0.4]])
    assert close_enough(np.ones(3) @ A, np.ones(3))
    state = np.array([1.0, 0.0, 0.0])
    history = [state]
    for _ in range(18):
        state = A @ state
        history.append(state)
    history = np.asarray(history)
    steady = np.array([0.3, 0.3, 0.4])
    assert np.linalg.norm(history[-1] - steady) < 1e-10
    fig, (ax,) = new_figure()
    k = np.arange(len(history))
    for i, (color, label) in enumerate([(BLUE, r"$u_1$"), (ORANGE, r"$u_2$"), (GREEN, r"$u_3$")]):
        ax.plot(k, history[:, i], color=color, linewidth=2.2, marker="o", markersize=3, label=label)
        ax.axhline(steady[i], color=color, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(7.0, 0.84, r"$Au_\infty=u_\infty,\quad u_\infty=(0.3,0.3,0.4)^T$", color=DARK, fontsize=10)
    style_axis(ax, r"$k$", "mass")
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.03, 1.04)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("This positive column-stochastic matrix converges to its steady state", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit2-markov-steady-state")


def unit2_fourier_basis():
    x = np.linspace(-np.pi, np.pi, 700)
    f1 = np.sin(x)
    f3 = 0.4 * np.cos(3 * x)
    signal = f1 + f3
    inner = np.trapz(np.sin(x) * np.cos(3 * x), x)
    assert abs(inner) < 1e-8
    fig, (ax, bx) = new_figure(2)
    ax.plot(x, f1, color=BLUE, linewidth=2.1, label=r"$\sin t$")
    ax.plot(x, f3, color=ORANGE, linewidth=2.1, label=r"$0.4\cos 3t$")
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.set_title("orthogonal modes", color=DARK, fontsize=11)
    style_axis(ax, r"$t$", "amplitude")
    ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.legend(frameon=False, fontsize=8)
    bx.plot(x, signal, color=GREEN, linewidth=2.5)
    bx.axhline(0, color=DARK, linewidth=0.8)
    bx.set_title(r"signal $f(t)=\sin t+0.4\cos 3t$", color=DARK, fontsize=11)
    style_axis(bx, r"$t$", "amplitude")
    bx.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    fig.suptitle("Fourier series is a change to an orthogonal function basis", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "mit18.06sc-unit2-fourier-basis")


def unit3_spectral_positive_definite():
    angle = np.deg2rad(28)
    Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    eigenvalues = np.array([4.0, 1.0])
    A = Q @ np.diag(eigenvalues) @ Q.T
    assert close_enough(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    theta = np.linspace(0, 2 * np.pi, 500)
    unit = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = Q @ np.diag(1 / np.sqrt(eigenvalues)) @ unit
    values = np.einsum("ij,ji->i", ellipse.T @ A, ellipse)
    assert np.max(np.abs(values - 1)) < 1e-9
    fig, (ax,) = new_figure()
    ax.fill(ellipse[0], ellipse[1], color=PALE_GREEN, alpha=0.9)
    ax.plot(ellipse[0], ellipse[1], color=GREEN, linewidth=2.5)
    q1, q2 = Q[:, 0], Q[:, 1]
    vector(ax, 0.5 * q1, RED, r"$q_1,\lambda_1=4$")
    vector(ax, 1.0 * q2, BLUE, r"$q_2,\lambda_2=1$")
    ax.text(-1.35, 1.25, r"$x^TAx=1$", color=DARK, fontsize=12)
    ax.text(-1.35, 1.00, r"$A=Q\Lambda Q^T$", color=DARK, fontsize=10)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.55)
    ax.set_title("A positive definite quadratic form has eigenvector axes", color=DARK, fontsize=13)
    fig.tight_layout()
    save(fig, "mit18.06sc-unit3-spectral-positive-definite")


def unit3_quadratic_bowl():
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_subplot(121, projection="3d")
    bx = fig.add_subplot(122)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = 3 * X**2 + 2 * X * Y + 2 * Y**2
    assert np.min(Z) >= 0
    ax.plot_surface(X, Y, Z, cmap="Blues", linewidth=0, alpha=0.9, antialiased=True)
    ax.scatter([0], [0], [0], color=RED, s=35)
    ax.set_xlabel(r"$x_1$", fontsize=8)
    ax.set_ylabel(r"$x_2$", fontsize=8)
    ax.set_zlabel(r"$x^TAx$", fontsize=8)
    ax.set_title("quadratic bowl", color=DARK, fontsize=11)
    contours = bx.contour(X, Y, Z, levels=[1, 3, 6, 10, 16], colors=[BLUE, GREEN, ORANGE, PURPLE, RED])
    bx.clabel(contours, fontsize=7)
    bx.scatter([0], [0], color=RED, s=35)
    bx.text(0.12, 0.12, "unique minimum", color=RED, fontsize=9)
    style_axis(bx)
    bx.set_aspect("equal")
    bx.set_title(r"level sets of $x^TAx$", color=DARK, fontsize=11)
    fig.suptitle(r"Positive definiteness means every nonzero direction rises", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    save(fig, "mit18.06sc-unit3-quadratic-bowl")


def unit3_complex_action():
    radius = 1.55
    angle = np.deg2rad(42)
    alpha = radius * np.exp(1j * angle)
    z = 0.9 * np.exp(1j * np.deg2rad(18))
    image = alpha * z
    assert close_enough(abs(image), abs(alpha) * abs(z))
    assert close_enough(np.angle(image), np.angle(alpha) + np.angle(z))
    theta = np.linspace(0, 2 * np.pi, 400)
    fig, (ax, bx) = new_figure(2)
    ax.plot(np.cos(theta), np.sin(theta), color=LIGHT, linewidth=2)
    vector(ax, [z.real, z.imag], BLUE, r"$z$")
    ax.set_title("complex input", color=DARK, fontsize=11)
    style_axis(ax, r"$\Re z$", r"$\Im z$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    bx.plot(radius * np.cos(theta), radius * np.sin(theta), color=PALE_ORANGE, linewidth=2)
    vector(bx, [image.real, image.imag], ORANGE, r"$\alpha z$")
    bx.set_title("rotate and scale", color=DARK, fontsize=11)
    style_axis(bx, r"$\Re(\alpha z)$", r"$\Im(\alpha z)$")
    bx.set_aspect("equal")
    bx.set_xlim(-2.0, 2.0)
    bx.set_ylim(-2.0, 2.0)
    fig.suptitle(r"Multiplication by $\alpha=re^{i\theta}$ scales by $r$ and rotates by $\theta$", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "mit18.06sc-unit3-complex-action")


def unit3_fft_butterfly():
    n = 8
    stages = 4
    fig, (ax,) = new_figure()
    ax.set_xlim(-0.4, stages - 0.6)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.axis("off")
    for stage in range(stages):
        for i in range(n):
            ax.scatter(stage, i, s=24, color=BLUE if stage in (0, stages - 1) else DARK, zorder=5)
    distances = [1, 2, 4]
    for stage, distance in enumerate(distances):
        block = 2 * distance
        for start in range(0, n, block):
            for offset in range(distance):
                a = start + offset
                b = a + distance
                for src in (a, b):
                    ax.plot([stage, stage + 1], [src, a], color=BLUE, linewidth=1.0, alpha=0.8)
                    ax.plot([stage, stage + 1], [src, b], color=ORANGE, linewidth=1.0, alpha=0.8)
    bit_count = int(np.log2(n))
    bit_reversed = [int(f"{i:0{bit_count}b}"[::-1], 2) for i in range(n)]
    assert sorted(bit_reversed) == list(range(n))
    for row, input_index in enumerate(bit_reversed):
        ax.text(-0.12, row, rf"$x_{{{input_index}}}$", ha="right", va="center", color=DARK, fontsize=8)
    for output_index in range(n):
        ax.text(stages - 0.88, output_index, rf"$X_{{{output_index}}}$", ha="left", va="center", color=DARK, fontsize=8)
    for stage, label in enumerate(["bit-reversed input", r"$m=2$", r"$m=4$", r"$m=8$"]):
        ax.text(stage, -0.42, label, ha="center", color=MID, fontsize=8)
    assert sum(1 for _ in distances) == int(np.log2(n))
    ax.text(1.5, 7.65, r"each butterfly combines $a$ and $\omega^k b$ as $a\pm\omega^k b$", ha="center", color=DARK, fontsize=9)
    ax.set_title(r"Radix-2 FFT: $O(n\log_2 n)$ structured combinations", color=DARK, fontsize=13, pad=14)
    save(fig, "mit18.06sc-unit3-fft-butterfly")


def unit3_jordan_chain():
    A = np.array([[2.0, 1.0], [0.0, 2.0]])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    N = A - 2 * np.eye(2)
    assert close_enough(N @ v1, [0, 0])
    assert close_enough(N @ v2, v1)
    k = 5
    assert close_enough(np.linalg.matrix_power(A, k) @ v2, (2**k) * v2 + k * (2 ** (k - 1)) * v1)
    fig, (ax,) = new_figure()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    positions = [(1.0, 2.0), (4.2, 2.0), (7.4, 2.0)]
    labels = [r"generalized vector $v_2$", r"eigenvector $v_1$", r"$0$"]
    colors = [PALE_ORANGE, PALE_BLUE, PALE_GREEN]
    for (x, y), label, face in zip(positions, labels, colors):
        ax.add_patch(Rectangle((x, y), 1.8, 1.0, facecolor=face, edgecolor=DARK, linewidth=1.2))
        ax.text(x + 0.9, y + 0.5, label, ha="center", va="center", color=DARK, fontsize=9)
    arrow(ax, (2.8, 2.5), (4.2, 2.5), ORANGE, 2.0)
    arrow(ax, (6.0, 2.5), (7.4, 2.5), BLUE, 2.0)
    ax.text(3.5, 2.82, r"$A-2I$", ha="center", color=ORANGE, fontsize=10)
    ax.text(6.7, 2.82, r"$A-2I$", ha="center", color=BLUE, fontsize=10)
    ax.text(5.0, 4.2, r"$(A-2I)v_2=v_1,\qquad (A-2I)v_1=0$", ha="center", color=DARK, fontsize=12)
    ax.text(5.0, 0.85, r"$A^kv_2=2^kv_2+k\,2^{k-1}v_1$", ha="center", color=RED, fontsize=11)
    ax.set_title("A Jordan block adds polynomial growth to exponential growth", color=DARK, fontsize=13)
    save(fig, "mit18.06sc-unit3-jordan-chain")


def unit3_svd_geometry():
    phi = np.deg2rad(32)
    psi = np.deg2rad(-18)
    V = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    U = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    sigma = np.array([2.4, 0.8])
    A = U @ np.diag(sigma) @ V.T
    assert close_enough(A @ V[:, 0], sigma[0] * U[:, 0])
    assert close_enough(A @ V[:, 1], sigma[1] * U[:, 1])
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    image = A @ circle
    fig, (ax, bx) = new_figure(2)
    ax.plot(circle[0], circle[1], color=BLUE, linewidth=2.4)
    vector(ax, V[:, 0], ORANGE, r"$v_1$")
    vector(ax, V[:, 1], GREEN, r"$v_2$")
    ax.set_title("unit circle in input", color=DARK, fontsize=11)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    bx.fill(image[0], image[1], color=PALE_GREEN, alpha=0.8)
    bx.plot(image[0], image[1], color=GREEN, linewidth=2.4)
    vector(bx, sigma[0] * U[:, 0], ORANGE, r"$\sigma_1u_1$")
    vector(bx, sigma[1] * U[:, 1], BLUE, r"$\sigma_2u_2$")
    bx.set_title("ellipse in output", color=DARK, fontsize=11)
    style_axis(bx, r"$y_1$", r"$y_2$")
    bx.set_aspect("equal")
    bx.set_xlim(-2.8, 2.8)
    bx.set_ylim(-2.1, 2.1)
    fig.suptitle(r"SVD: $Av_i=\sigma_i u_i$", color=DARK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "mit18.06sc-unit3-svd-geometry")


def unit3_change_of_basis():
    B = np.array([[1.0, 1.0], [0.0, 2.0]])
    coordinates = np.array([2.0, 1.0])
    x = B @ coordinates
    assert close_enough(x, [3, 2])
    fig, (ax, bx) = new_figure(2)
    b1, b2 = B[:, 0], B[:, 1]
    vector(ax, b1, BLUE, r"$b_1$")
    vector(ax, b2, ORANGE, r"$b_2$")
    vector(ax, x, GREEN, r"$x=2b_1+b_2$")
    ax.plot([2 * b1[0], x[0]], [2 * b1[1], x[1]], color=ORANGE, linestyle="--")
    ax.plot([b2[0], x[0]], [b2[1], x[1]], color=BLUE, linestyle="--")
    ax.set_title("geometric vector", color=DARK, fontsize=11)
    style_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 3.8)
    ax.set_ylim(-0.6, 3.2)
    bx.scatter([coordinates[0]], [coordinates[1]], color=GREEN, s=55)
    arrow(bx, (0, 0), coordinates, GREEN, 2.0)
    bx.text(coordinates[0] + 0.1, coordinates[1] + 0.1, r"$[x]_B=(2,1)$", color=GREEN, fontsize=10)
    bx.set_title("coordinates in basis B", color=DARK, fontsize=11)
    style_axis(bx, r"coefficient of $b_1$", r"coefficient of $b_2$")
    bx.set_aspect("equal")
    bx.set_xlim(-0.6, 3.2)
    bx.set_ylim(-0.6, 2.6)
    fig.suptitle(r"The vector is unchanged; only its coordinate description changes: $x=B[x]_B$", color=DARK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "mit18.06sc-unit3-change-of-basis")


def unit3_pseudoinverse():
    u = np.array([1.0, 1.0]) / np.sqrt(2)
    v = np.array([1.0, 2.0]) / np.sqrt(5)
    sigma = 2.0
    A = sigma * np.outer(u, v)
    A_plus = np.outer(v, u) / sigma
    b = np.array([1.2, 0.2])
    p = u * (u @ b)
    e = b - p
    xhat = A_plus @ b
    n = np.array([-v[1], v[0]])
    assert close_enough(A @ xhat, p)
    assert close_enough(u @ e, 0)
    assert close_enough(v @ n, 0)
    assert close_enough(A @ A_plus @ A, A)
    fig, (ax, bx) = new_figure(2)
    t = np.linspace(-1.7, 1.7, 100)
    ax.plot(t * u[0], t * u[1], color=BLUE, linewidth=2.4, label=r"$C(A)$")
    vector(ax, b, ORANGE)
    vector(ax, p, GREEN)
    arrow(ax, p, b, RED, 1.8)
    ax.text(1.27, 0.05, r"$b$", color=ORANGE, fontsize=9)
    ax.text(0.43, 0.82, r"$p=AA^+b$", color=GREEN, fontsize=9)
    ax.text(1.02, 0.48, r"$e=b-p\in N(A^T)$", color=RED, fontsize=8)
    ax.set_title("output space", color=DARK, fontsize=11)
    style_axis(ax, r"$y_1$", r"$y_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.8)
    ax.set_ylim(-1.4, 1.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    bx.plot(t * n[0], t * n[1], color=ORANGE, linewidth=2.4, label=r"$N(A)$")
    bx.plot(t * v[0], t * v[1], color=BLUE, linewidth=2.4, label=r"$N(A)^\perp$")
    vector(bx, xhat, GREEN, r"$\hat x=A^+b$")
    bx.set_title("input space: minimum-norm solution", color=DARK, fontsize=11)
    style_axis(bx)
    bx.set_aspect("equal")
    bx.set_xlim(-1.6, 1.8)
    bx.set_ylim(-1.4, 1.8)
    bx.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(r"The pseudoinverse projects first, then returns the shortest input", color=DARK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "mit18.06sc-unit3-pseudoinverse")


FIGURES = [
    unit1_row_column_picture,
    unit1_elimination_lu,
    unit1_affine_solution,
    unit1_four_subspaces,
    unit1_rank_one,
    unit1_incidence_network,
    unit2_orthogonal_complements,
    unit2_projection,
    unit2_least_squares_fit,
    unit2_gram_schmidt,
    unit2_determinant_volume,
    unit2_eigenvectors,
    unit2_diagonalization,
    unit2_matrix_exponential,
    unit2_markov_steady_state,
    unit2_fourier_basis,
    unit3_spectral_positive_definite,
    unit3_quadratic_bowl,
    unit3_complex_action,
    unit3_fft_butterfly,
    unit3_jordan_chain,
    unit3_svd_geometry,
    unit3_change_of_basis,
    unit3_pseudoinverse,
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for draw in FIGURES:
        draw()
    expected = {f"{draw.__name__.replace('_', '-')}.png" for draw in []}
    produced = sorted(OUTPUT.glob("mit18.06sc-unit*.png"))
    if len(produced) != len(FIGURES):
        raise RuntimeError(f"Expected {len(FIGURES)} PNG files, found {len(produced)}")
    for path in produced:
        if path.stat().st_size < 10_000:
            raise RuntimeError(f"Generated figure is unexpectedly small: {path}")
    print(f"Generated {len(produced)} figures in {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Figure generation failed: {exc}", file=sys.stderr)
        raise
