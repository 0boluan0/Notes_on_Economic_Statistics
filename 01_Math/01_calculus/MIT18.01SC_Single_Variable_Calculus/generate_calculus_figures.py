#!/usr/bin/env python3
"""Generate the 25 deterministic teaching figures used by the MIT 18.01SC notes."""

from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("Install numpy and matplotlib before running this script.") from exc


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "98_attachment" / "MIT18.01SC"

BLUE = "#2563A6"
ORANGE = "#D97706"
GREEN = "#2E8B57"
RED = "#C43D3D"
DARK = "#293241"
LIGHT = "#D9E2EC"
PALE_BLUE = "#DCEBFA"
PALE_GREEN = "#DDF2E5"


def close_enough(a, b, tolerance=1e-9):
    return np.allclose(a, b, rtol=tolerance, atol=tolerance)


def new_figure(columns=1):
    fig, axes = plt.subplots(1, columns, figsize=(8, 4.5), dpi=200)
    return fig, np.atleast_1d(axes)


def style_axis(ax, x_label="x", y_label="y", grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(DARK)
    ax.tick_params(colors=DARK, labelsize=8)
    ax.set_xlabel(x_label, color=DARK, fontsize=10)
    ax.set_ylabel(y_label, color=DARK, fontsize=10, rotation=0, labelpad=10)
    if grid:
        ax.grid(True, color=LIGHT, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def arrow(ax, start, end, color=DARK, width=1.5, mutation=11):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=mutation,
                                linewidth=width, color=color))


def save(fig, stem):
    fig.patch.set_facecolor("white")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=200, facecolor="white",
                bbox_inches=None, pad_inches=0.08)
    plt.close(fig)


def unit01_secant_tangent():
    fig, (ax,) = new_figure()
    f = lambda x: 0.55 * x**2 + 0.35
    a, b = 0.8, 2.25
    fa, fb = f(a), f(b)
    secant_slope = (fb - fa) / (b - a)
    tangent_slope = 1.1 * a
    assert close_enough(secant_slope, 0.55 * (a + b))
    x = np.linspace(-0.35, 3.0, 500)
    ax.plot(x, f(x), color=BLUE, linewidth=2.6, label=r"$y=f(x)$")
    ax.plot(x, fa + secant_slope * (x - a), color=ORANGE, linewidth=2,
            label="secant through P and Q")
    ax.plot(x, fa + tangent_slope * (x - a), color=GREEN, linewidth=2,
            linestyle="--", label="tangent at P")
    ax.scatter([a, b], [fa, fb], s=35, color=[GREEN, ORANGE], zorder=5)
    ax.annotate(r"$P=(a,f(a))$", (a, fa), xytext=(-80, 18), textcoords="offset points",
                color=GREEN, fontsize=9, arrowprops=dict(arrowstyle="-", color=GREEN))
    ax.annotate(r"$Q=(a+h,f(a+h))$", (b, fb), xytext=(8, 15), textcoords="offset points",
                color=ORANGE, fontsize=9)
    ax.text(1.38, 1.08, r"$m_{PQ}=dfrac{f(a+h)-f(a)}{h}$", color=ORANGE, fontsize=10)
    ax.text(0.03, 3.45, r"$Q\to P\quad\Longrightarrow\quad m_{PQ}\to f'(a)$",
            color=DARK, fontsize=11)
    style_axis(ax)
    ax.set_xlim(-0.35, 3.0); ax.set_ylim(-0.1, 5.45)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "unit01-secant-tangent")


def unit01_trig_squeeze():
    fig, (ax,) = new_figure()
    theta = 0.72
    p = np.array([np.cos(theta), np.sin(theta)])
    t = np.array([1.0, np.tan(theta)])
    assert 0 < np.sin(theta) < theta < np.tan(theta)
    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=105, color=BLUE, linewidth=2.3))
    ax.plot([0, 1.2], [0, 0], color=DARK, linewidth=1.3)
    ax.plot([0, p[0]], [0, p[1]], color=BLUE, linewidth=2)
    ax.plot([0, t[0]], [0, t[1]], color=ORANGE, linewidth=2)
    ax.plot([p[0], p[0]], [0, p[1]], color=GREEN, linewidth=1.8)
    ax.plot([1, 1], [0, t[1]], color=ORANGE, linewidth=1.8)
    ax.add_patch(Wedge((0, 0), 1, 0, np.degrees(theta), facecolor=PALE_BLUE,
                       edgecolor="none", alpha=0.75))
    ax.add_patch(Arc((0, 0), 0.42, 0.42, theta1=0, theta2=np.degrees(theta),
                     color=DARK, linewidth=1.2))
    ax.scatter(*p, color=BLUE, s=28, zorder=4)
    ax.text(0.24, 0.08, r"$\theta$", fontsize=11, color=DARK)
    ax.text(p[0] - 0.25, p[1] + 0.08, r"$(\cos\theta,\sin\theta)$", fontsize=9, color=BLUE)
    ax.text(1.03, t[1] - 0.04, r"$\tan\theta$", fontsize=9, color=ORANGE)
    ax.text(p[0] - 0.13, p[1] / 2, r"$\sin\theta$", fontsize=9, color=GREEN,
            rotation=90, va="center")
    ax.text(-1.15, -0.25,
            r"$\frac{1}{2}\sin\theta<\frac{1}{2}\theta<\frac{1}{2}\tan\theta$",
            fontsize=11, color=DARK)
    ax.text(-1.15, -0.39,
            r"$\Rightarrow\quad \cos\theta<\dfrac{\sin\theta}{\theta}<1$",
            fontsize=11, color=DARK)
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.55); ax.set_ylim(-0.42, 1.35)
    ax.axis("off")
    save(fig, "unit01-trig-squeeze")


def unit01_inverse_reflection():
    fig, (ax,) = new_figure()
    f = lambda x: 0.42 * x**2 + 0.65 * x + 0.3
    x = np.linspace(0, 2.5, 400); y = f(x)
    ax.plot(x, y, color=BLUE, linewidth=2.5, label=r"$y=f(x)$")
    ax.plot(y, x, color=ORANGE, linewidth=2.5, label=r"$y=f^{-1}(x)$")
    diagonal = np.linspace(0, 4.6, 100)
    ax.plot(diagonal, diagonal, color=DARK, linestyle="--", linewidth=1.4, label=r"$y=x$")
    a = 1.25; b = f(a); m = 0.84 * a + 0.65
    assert m != 0 and close_enough((1 / m) * m, 1)
    u = np.linspace(a - 0.7, a + 0.7, 80)
    ax.plot(u, b + m * (u - a), color=GREEN, linewidth=1.8)
    v = np.linspace(b - 0.9, b + 0.9, 80)
    ax.plot(v, a + (v - b) / m, color=RED, linewidth=1.8)
    ax.scatter([a, b], [b, a], color=[GREEN, RED], s=35, zorder=5)
    ax.text(0.14, 4.1, r"$(f^{-1})'(b)=\dfrac{1}{f'(a)},\quad b=f(a)$", fontsize=11, color=DARK)
    style_axis(ax); ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 4.7); ax.set_ylim(0, 4.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "unit01-inverse-reflection")


def unit01_exp_log():
    fig, (ax,) = new_figure()
    x_exp = np.linspace(-2.2, 1.65, 500)
    x_log = np.linspace(0.08, 5.1, 500)
    ax.plot(x_exp, np.exp(x_exp), color=BLUE, linewidth=2.5, label=r"$y=e^x$")
    ax.plot(x_log, np.log(x_log), color=ORANGE, linewidth=2.5, label=r"$y=\ln x$")
    d = np.linspace(-2.2, 5.1, 100)
    ax.plot(d, d, color=DARK, linestyle="--", linewidth=1.2, label=r"$y=x$")
    ax.scatter([0, 1], [1, 0], color=[BLUE, ORANGE], s=38, zorder=4)
    ax.plot(np.linspace(-1.3, 1.2, 80), 1 + np.linspace(-1.3, 1.2, 80),
            color=GREEN, linewidth=1.7)
    assert close_enough(np.exp(np.log(x_log)), x_log)
    ax.text(-2.0, 4.55, r"$e^{\ln x}=x,\qquad \ln(e^x)=x$", fontsize=11, color=DARK)
    ax.text(-1.55, 0.08, r"tangent to $e^x$ at $x=0$: $y=1+x$", fontsize=9, color=GREEN)
    style_axis(ax); ax.set_xlim(-2.2, 5.2); ax.set_ylim(-2.1, 5.2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "unit01-exp-log")


def unit01_hyperbolic():
    fig, axes = new_figure(2)
    ax, bx = axes
    t = np.linspace(-1.7, 1.7, 400)
    x, y = np.cosh(t), np.sinh(t)
    assert np.max(np.abs(x**2 - y**2 - 1)) < 1e-10
    ax.plot(x, y, color=BLUE, linewidth=2.6)
    ax.plot(-x, y, color=LIGHT, linewidth=1.4)
    asym = np.linspace(-3, 3, 100)
    ax.plot(asym, asym, color=DARK, linestyle="--", linewidth=1)
    ax.plot(asym, -asym, color=DARK, linestyle="--", linewidth=1)
    point_t = 1.0; point = (np.cosh(point_t), np.sinh(point_t))
    ax.scatter(*point, color=ORANGE, s=35)
    ax.text(point[0] + 0.08, point[1], r"$(\cosh t,\sinh t)$", fontsize=8, color=ORANGE)
    ax.text(-2.65, 2.55, r"$x^2-y^2=1$", fontsize=11, color=DARK)
    style_axis(ax, grid=False); ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    phi = np.linspace(0, 2 * np.pi, 400)
    bx.plot(np.cos(phi), np.sin(phi), color=GREEN, linewidth=2.6)
    bx.text(-1.25, 1.2, r"$x^2+y^2=1$", fontsize=11, color=DARK)
    bx.text(-1.36, -1.43, "circle: trigonometric\nhyperbola: hyperbolic", fontsize=9, color=DARK)
    style_axis(bx, grid=False); bx.set_aspect("equal"); bx.set_xlim(-1.55, 1.55); bx.set_ylim(-1.55, 1.55)
    save(fig, "unit01-hyperbolic")


def unit02_linear_quadratic():
    fig, (ax,) = new_figure()
    f = np.exp; a = 0.0
    x = np.linspace(-1.6, 1.35, 500)
    p1 = 1 + x; p2 = 1 + x + x**2 / 2
    assert close_enough([f(a), f(a), f(a)], [1, p1[np.argmin(abs(x-a))], p2[np.argmin(abs(x-a))]], 2e-2)
    ax.plot(x, f(x), color=BLUE, linewidth=2.7, label=r"$e^x$")
    ax.plot(x, p1, color=ORANGE, linewidth=2, label=r"$P_1(x)=1+x$")
    ax.plot(x, p2, color=GREEN, linewidth=2, label=r"$P_2(x)=1+x+x^2/2$")
    ax.scatter([0], [1], color=DARK, s=35, zorder=5)
    ax.axvspan(-0.55, 0.55, color=PALE_BLUE, alpha=0.4)
    ax.text(-1.48, 3.65, r"At $a=0$: value, slope, and curvature are matched in order.", fontsize=9, color=DARK)
    style_axis(ax); ax.set_xlim(-1.6, 1.35); ax.set_ylim(-0.45, 4.2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "unit02-linear-quadratic")


def unit02_curve_sign_chart():
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), dpi=200, gridspec_kw={"height_ratios": [3, 1]})
    ax, bx = axes
    f = lambda x: 3*x - x**3
    x = np.linspace(-2.15, 2.15, 600)
    ax.plot(x, f(x), color=BLUE, linewidth=2.6)
    ax.scatter([-1, 0, 1], [f(-1), f(0), f(1)], color=[ORANGE, GREEN, ORANGE], s=36, zorder=5)
    ax.axvline(0, color=GREEN, linestyle="--", linewidth=1.2)
    assert close_enough([3 - 3*(-1)**2, 3 - 3*(1)**2], [0, 0])
    ax.text(-1.42, -2.65, "local min", fontsize=9, color=ORANGE)
    ax.text(1.04, 2.25, "local max", fontsize=9, color=ORANGE)
    ax.text(0.06, -0.65, "inflection", fontsize=9, color=GREEN)
    style_axis(ax); ax.set_xlim(-2.15, 2.15); ax.set_ylim(-3.7, 3.7)
    bx.axhline(0, color=DARK, linewidth=1.2)
    for value in (-1, 0, 1): bx.axvline(value, color=LIGHT, linewidth=1)
    bx.text(-1.65, 0.32, r"$f'<0$", color=RED, ha="center")
    bx.text(-0.5, 0.32, r"$f'>0$", color=GREEN, ha="center")
    bx.text(0.5, 0.32, r"$f'>0$", color=GREEN, ha="center")
    bx.text(1.65, 0.32, r"$f'<0$", color=RED, ha="center")
    bx.text(-0.55, -0.45, r"$f''>0$", color=GREEN, ha="center")
    bx.text(0.55, -0.45, r"$f''<0$", color=RED, ha="center")
    bx.set_xlim(-2.15, 2.15); bx.set_ylim(-0.8, 0.8); bx.axis("off")
    save(fig, "unit02-curve-sign-chart")


def unit02_related_rates_cone():
    fig, axes = new_figure(2)
    ax, bx = axes
    H, R, h = 4.0, 2.2, 2.5; r = R * h / H
    assert close_enough(r / h, R / H)
    cone = Polygon([(-R, H), (0, 0), (R, H)], closed=False, fill=False, edgecolor=DARK, linewidth=2.2)
    ax.add_patch(cone)
    water = Polygon([(-r, h), (0, 0), (r, h)], closed=True, facecolor=PALE_BLUE,
                    edgecolor=BLUE, linewidth=1.6, alpha=0.85)
    ax.add_patch(water)
    ax.plot([-r, r], [h, h], color=BLUE, linewidth=2)
    arrow(ax, (0.15, 0), (0.15, h), GREEN); ax.text(0.25, h/2, r"$h(t)$", color=GREEN, fontsize=10)
    arrow(ax, (0, h+0.12), (r, h+0.12), ORANGE); ax.text(r/2, h+0.28, r"$r(t)$", color=ORANGE, fontsize=10)
    ax.text(-2.0, 4.25, r"$\dfrac{r}{h}=\dfrac{R}{H}$", fontsize=12, color=DARK)
    ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-0.3, 4.8); ax.axis("off")
    bx.text(0.05, 0.82, "Similar triangles", fontsize=12, color=DARK, weight="bold")
    bx.text(0.05, 0.64, r"$r=\dfrac{R}{H}h$", fontsize=15, color=BLUE)
    bx.text(0.05, 0.45, r"$V=\dfrac{1}{3}\pi r^2h=\dfrac{\pi R^2}{3H^2}h^3$", fontsize=12, color=DARK)
    bx.text(0.05, 0.25, r"$\dfrac{dV}{dt}=\dfrac{\pi R^2}{H^2}h^2\dfrac{dh}{dt}$", fontsize=12, color=GREEN)
    bx.text(0.05, 0.08, "Differentiate only after reducing to one variable.", fontsize=9, color=RED)
    bx.axis("off")
    save(fig, "unit02-related-rates-cone")


def unit02_newton_method():
    fig, (ax,) = new_figure()
    f = lambda x: x**2 - 2
    fp = lambda x: 2*x
    values = [2.6]
    for _ in range(3): values.append(values[-1] - f(values[-1]) / fp(values[-1]))
    assert all(abs(f(values[i+1])) < abs(f(values[i])) for i in range(3))
    x = np.linspace(0.7, 2.8, 500)
    ax.plot(x, f(x), color=BLUE, linewidth=2.6, label=r"$f(x)=x^2-2$")
    ax.axhline(0, color=DARK, linewidth=1.2)
    colors = [ORANGE, GREEN, RED]
    for index, (x0, x1) in enumerate(zip(values[:-1], values[1:])):
        tx = np.linspace(x1, x0 + 0.08, 80)
        ax.plot(tx, f(x0) + fp(x0)*(tx-x0), color=colors[index], linewidth=1.7)
        ax.plot([x0, x0], [0, f(x0)], color=colors[index], linestyle=":", linewidth=1.2)
        ax.scatter([x0, x1], [f(x0), 0], color=colors[index], s=27, zorder=5)
        ax.text(x0 + 0.025, f(x0) + 0.12, rf"$x_{index}$", color=colors[index], fontsize=9)
    ax.text(0.78, 4.8, r"$x_{n+1}=x_n-\dfrac{f(x_n)}{f'(x_n)}$", fontsize=12, color=DARK)
    ax.text(1.48, -0.65, r"$\sqrt{2}$", fontsize=10, color=BLUE)
    style_axis(ax); ax.set_xlim(0.7, 2.8); ax.set_ylim(-1.0, 5.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "unit02-newton-method")


def unit02_mean_value_theorem():
    fig, (ax,) = new_figure()
    f = lambda x: 0.32*x**3 - 0.55*x + 1.2
    fp = lambda x: 0.96*x**2 - 0.55
    a, b = -1.2, 1.65
    secant = (f(b)-f(a))/(b-a)
    c = np.sqrt((secant + 0.55)/0.96)
    assert a < c < b and close_enough(fp(c), secant)
    x = np.linspace(-1.7, 2.05, 500)
    ax.plot(x, f(x), color=BLUE, linewidth=2.6, label=r"$y=f(x)$")
    ax.plot(x, f(a)+secant*(x-a), color=ORANGE, linewidth=2, label="secant AB")
    ax.plot(x, f(c)+secant*(x-c), color=GREEN, linestyle="--", linewidth=2,
            label="parallel tangent at c")
    ax.scatter([a,b,c], [f(a),f(b),f(c)], color=[ORANGE,ORANGE,GREEN], s=35, zorder=5)
    ax.text(a-0.18, f(a)+0.12, "A", color=ORANGE); ax.text(b+0.05, f(b), "B", color=ORANGE)
    ax.text(c+0.05, f(c)-0.25, "c", color=GREEN)
    ax.text(-1.55, 3.05, r"$f'(c)=\dfrac{f(b)-f(a)}{b-a}$", fontsize=12, color=DARK)
    style_axis(ax); ax.set_xlim(-1.7,2.05); ax.set_ylim(-0.1,3.55)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "unit02-mean-value-theorem")


def unit02_separation_of_variables():
    fig, (ax,) = new_figure()
    ax.axis("off")
    boxes = [
        (0.03, 0.7, 0.2, 0.16, r"$\dfrac{dy}{dx}=g(x)h(y)$"),
        (0.29, 0.7, 0.2, 0.16, r"$\dfrac{dy}{h(y)}=g(x)\,dx$"),
        (0.55, 0.7, 0.2, 0.16, r"$\int\dfrac{dy}{h(y)}=\int g(x)\,dx$"),
        (0.77, 0.7, 0.2, 0.16, r"$H(y)=G(x)+C$"),
        (0.29, 0.3, 0.2, 0.16, r"Solve for $y$"),
        (0.55, 0.3, 0.2, 0.16, r"Use initial data"),
        (0.77, 0.3, 0.21, 0.16, "Check domain\nand equilibria"),
    ]
    for i, (x, y, w, h, label) in enumerate(boxes):
        color = BLUE if i < 4 else GREEN
        ax.add_patch(Rectangle((x,y), w,h, transform=ax.transAxes, facecolor="white",
                               edgecolor=color, linewidth=1.8))
        ax.text(x+w/2, y+h/2, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5 if i in (2, 6) else 9, color=DARK)
    for start, end in [((.23,.78),(.29,.78)),((.49,.78),(.55,.78)),((.75,.78),(.77,.78)),
                       ((.87,.70),(.42,.46)),((.49,.38),(.55,.38)),((.75,.38),(.77,.38))]:
        arrow(ax, start, end, ORANGE, 1.4, 10)
    ax.text(0.03, 0.12, r"Do not divide by $h(y)$ before recording solutions with $h(y)=0$.",
            transform=ax.transAxes, fontsize=10, color=RED)
    save(fig, "unit02-separation-of-variables")


def unit03_riemann_sums():
    fig, (ax,) = new_figure()
    f = lambda x: 1.0 + 0.45*x + 0.5*np.sin(1.4*x)
    a,b,n = 0.0,5.0,10; edges=np.linspace(a,b,n+1); dx=(b-a)/n
    mids=(edges[:-1]+edges[1:])/2; heights=f(mids)
    assert close_enough(np.diff(edges), dx)
    for left,height in zip(edges[:-1],heights):
        ax.add_patch(Rectangle((left,0),dx,height,facecolor=PALE_GREEN,edgecolor=GREEN,linewidth=1))
    x=np.linspace(a,b,600); ax.plot(x,f(x),color=BLUE,linewidth=2.7,label=r"$y=f(x)$")
    ax.scatter(mids,heights,color=ORANGE,s=12,zorder=4,label="midpoint samples")
    ax.text(0.2,3.45,r"$\sum_{i=1}^n f(x_i^*)\,\Delta x\quad\longrightarrow\quad\int_a^b f(x)\,dx$",
            fontsize=12,color=DARK)
    style_axis(ax); ax.set_xlim(0,5); ax.set_ylim(0,3.9); ax.legend(frameon=False,fontsize=8)
    save(fig,"unit03-riemann-sums")


def unit03_ftc_accumulator():
    fig,(ax,)=new_figure()
    f=lambda t:1.1+0.35*t+0.35*np.sin(1.5*t)
    t=np.linspace(0,5,600); x,h=2.45,0.52
    ax.plot(t,f(t),color=BLUE,linewidth=2.6,label=r"$y=f(t)$")
    ax.fill_between(t,0,f(t),where=t<=x,color=PALE_BLUE,alpha=.85,label=r"$A(x)$")
    mask=(t>=x)&(t<=x+h); ax.fill_between(t,0,f(t),where=mask,color=GREEN,alpha=.45,
                                          label=r"$A(x+h)-A(x)$")
    ax.axvline(x,color=DARK,linewidth=1); ax.axvline(x+h,color=DARK,linewidth=1)
    ax.text(x-.07,-.18,r"$x$",fontsize=10); ax.text(x+h-.12,-.18,r"$x+h$",fontsize=10)
    ax.text(2.7,2.65,r"$\dfrac{A(x+h)-A(x)}{h}\to f(x)$",fontsize=12,color=DARK)
    assert x < x+h
    style_axis(ax,"t"); ax.set_xlim(0,5); ax.set_ylim(0,3.25); ax.legend(frameon=False,fontsize=8,loc="upper left")
    save(fig,"unit03-ftc-accumulator")


def unit03_signed_area():
    fig,(ax,)=new_figure()
    x=np.linspace(-2,2,500); top=2.2-.25*x**2; bottom=.45+.15*x**2
    assert np.all(top>=bottom)
    ax.plot(x,top,color=BLUE,linewidth=2.5,label=r"$y=f(x)$")
    ax.plot(x,bottom,color=ORANGE,linewidth=2.5,label=r"$y=g(x)$")
    ax.fill_between(x,bottom,top,color=PALE_GREEN,alpha=.9)
    sample=.7
    arrow(ax,(sample,bottom[np.argmin(abs(x-sample))]),(sample,top[np.argmin(abs(x-sample))]),GREEN)
    ax.text(.78,1.35,r"$f(x)-g(x)$",color=GREEN,fontsize=10)
    ax.text(-1.75,2.52,r"$\mathrm{Area}=\int_a^b [f(x)-g(x)]\,dx$",fontsize=12,color=DARK)
    style_axis(ax); ax.set_xlim(-2.1,2.1); ax.set_ylim(0,2.85); ax.legend(frameon=False,fontsize=9)
    save(fig,"unit03-signed-area")


def unit03_volume_methods():
    fig,axes=new_figure(2); ax,bx=axes
    x=np.linspace(0,2.5,300); y=1.8-.38*x
    ax.plot(x,y,color=BLUE,linewidth=2.4); ax.fill_between(x,0,y,color=PALE_BLUE,alpha=.7)
    x0=.9; y0=1.8-.38*x0
    ax.add_patch(Rectangle((x0-.04,0),.08,y0,facecolor=GREEN,edgecolor=GREEN))
    ax.plot([-.08,2.7],[0,0],color=DARK,linewidth=1.2)
    ax.text(.15,2.05,"Washer: slices perpendicular",fontsize=10,color=DARK)
    ax.text(.97,y0/2,r"$R=f(x)$",fontsize=9,color=GREEN,rotation=90,va="center")
    ax.text(.15,.25,r"$dV=\pi R^2\,dx$",fontsize=11,color=GREEN)
    ax.set_xlim(-.15,2.75); ax.set_ylim(-.2,2.35); ax.axis("off")
    bx.plot(x,y,color=BLUE,linewidth=2.4); bx.fill_between(x,0,y,color=PALE_BLUE,alpha=.7)
    x1=1.45; y1=1.8-.38*x1
    bx.add_patch(Rectangle((x1-.035,0),.07,y1,facecolor=ORANGE,edgecolor=ORANGE))
    bx.text(.15,2.05,"Shell: slices parallel",fontsize=10,color=DARK)
    arrow(bx,(0,.12),(x1,.12),ORANGE); bx.text(.65,.2,r"radius $x$",fontsize=8,color=ORANGE)
    bx.text(x1+.08,y1/2,r"height $f(x)$",fontsize=8,color=ORANGE,rotation=90,va="center")
    bx.text(.15,.42,r"$dV=2\pi x f(x)\,dx$",fontsize=11,color=ORANGE)
    bx.set_xlim(-.15,2.75); bx.set_ylim(-.2,2.35); bx.axis("off")
    save(fig,"unit03-volume-methods")


def unit03_numerical_integration():
    fig,axes=new_figure(2); f=lambda x:1+.18*x+.55*np.sin(1.15*x)
    for ax,method in zip(axes,("Trapezoidal rule","Simpson's rule")):
        x=np.linspace(0,4,500); ax.plot(x,f(x),color=BLUE,linewidth=2.5)
        nodes=np.linspace(0,4,5); values=f(nodes)
        if method.startswith("Trap"):
            ax.plot(nodes,values,color=ORANGE,linewidth=2,marker="o",markersize=4)
            ax.fill_between(nodes,0,values,color="#FCE8C8",alpha=.65)
        else:
            for i in (0,2):
                coef=np.polyfit(nodes[i:i+3],values[i:i+3],2)
                xx=np.linspace(nodes[i],nodes[i+2],120); yy=np.polyval(coef,xx)
                assert close_enough(np.polyval(coef,nodes[i:i+3]),values[i:i+3])
                ax.plot(xx,yy,color=GREEN,linewidth=2); ax.fill_between(xx,0,yy,color=PALE_GREEN,alpha=.65)
            ax.scatter(nodes,values,color=GREEN,s=16,zorder=4)
        ax.text(.15,2.1,method,fontsize=11,color=DARK)
        style_axis(ax); ax.set_xlim(0,4); ax.set_ylim(0,2.45)
    axes[0].text(.15,.15,r"straight chords",fontsize=9,color=ORANGE)
    axes[1].text(.15,.15,r"quadratics through triples",fontsize=9,color=GREEN)
    save(fig,"unit03-numerical-integration")


def unit04_trig_substitution_triangles():
    fig,axes=plt.subplots(1,3,figsize=(8,4.5),dpi=200)
    specifications=[
        ((0,0),(3,0),(0,4),[r"$x$",r"$\sqrt{a^2-x^2}$",r"$a$"],r"$x=a\sin\theta$"),
        ((0,0),(3,0),(0,2),[r"$a$",r"$x$",r"$\sqrt{a^2+x^2}$"],r"$x=a\tan\theta$"),
        ((0,0),(3,0),(0,2),[r"$a$",r"$\sqrt{x^2-a^2}$",r"$x$"],r"$x=a\sec\theta$"),
    ]
    for ax,(o,p,q,labels,title) in zip(axes,specifications):
        ax.add_patch(Polygon([o,p,q],closed=True,fill=False,edgecolor=BLUE,linewidth=2.2))
        ax.add_patch(Rectangle((0,0),.24,.24,fill=False,edgecolor=DARK,linewidth=1))
        ax.add_patch(Arc((0,0),1.0,1.0,theta1=0,theta2=np.degrees(np.arctan2(q[1],p[0])),color=ORANGE))
        ax.text(.55,.16,r"$\theta$",color=ORANGE,fontsize=10)
        ax.text(1.5,-.35,labels[0],ha="center",fontsize=10); ax.text(-.25,q[1]/2,labels[1],ha="right",fontsize=9)
        ax.text(1.65,q[1]/2+.18,labels[2],fontsize=9,rotation=np.degrees(np.arctan2(q[1],p[0]))-8)
        ax.text(1.5,max(q[1],p[1])+.55,title,ha="center",fontsize=11,color=DARK)
        ax.set_aspect("equal"); ax.set_xlim(-.9,3.7); ax.set_ylim(-.7,5.0); ax.axis("off")
    save(fig,"unit04-trig-substitution-triangles")


def unit04_arc_length_element():
    fig,(ax,)=new_figure(); f=lambda x:.25*x**2+.45
    x=np.linspace(0,3.3,400); ax.plot(x,f(x),color=BLUE,linewidth=2.6)
    x0,x1=1.15,2.45; p=np.array([x0,f(x0)]); q=np.array([x1,f(x1)])
    ax.plot([p[0],q[0]],[p[1],p[1]],color=ORANGE,linewidth=2)
    ax.plot([q[0],q[0]],[p[1],q[1]],color=GREEN,linewidth=2)
    ax.plot([p[0],q[0]],[p[1],q[1]],color=DARK,linewidth=2.2)
    dx=x1-x0; dy=q[1]-p[1]; ds=np.hypot(dx,dy); assert close_enough(ds**2,dx**2+dy**2)
    ax.text((x0+x1)/2,p[1]-.23,r"$dx$",color=ORANGE,fontsize=11,ha="center")
    ax.text(q[0]+.08,(p[1]+q[1])/2,r"$dy$",color=GREEN,fontsize=11)
    ax.text((x0+x1)/2-.08,(p[1]+q[1])/2+.12,r"$ds$",color=DARK,fontsize=11,rotation=24)
    ax.text(.15,3.25,r"$ds=\sqrt{dx^2+dy^2}=\sqrt{1+(y')^2}\,dx$",fontsize=12,color=DARK)
    style_axis(ax); ax.set_xlim(0,3.4); ax.set_ylim(0,3.75)
    save(fig,"unit04-arc-length-element")


def unit04_surface_of_revolution():
    fig,(ax,)=new_figure(); x=np.linspace(.3,3.4,300); y=.65+.34*x
    ax.plot(x,y,color=BLUE,linewidth=2.6); ax.plot(x,-y,color=BLUE,linewidth=1.2,alpha=.45)
    ax.axhline(0,color=DARK,linewidth=1.1)
    for xi in np.linspace(.45,3.25,8):
        yi=.65+.34*xi
        ellipse_x=.16*np.cos(np.linspace(0,2*np.pi,150))+xi
        ellipse_y=yi*np.sin(np.linspace(0,2*np.pi,150))
        ax.plot(ellipse_x,ellipse_y,color=LIGHT,linewidth=.8)
    x0,x1=1.75,2.08; y0=.65+.34*x0; y1=.65+.34*x1
    ax.plot([x0,x1],[y0,y1],color=ORANGE,linewidth=5,solid_capstyle="butt")
    arrow(ax,(x0,0),(x0,y0),GREEN); ax.text(x0+.08,y0/2,r"$y$",color=GREEN,fontsize=10)
    ax.text(2.0,1.68,r"$ds$",color=ORANGE,fontsize=10)
    ax.text(.35,2.15,r"$dS=2\pi y\,ds$",fontsize=13,color=DARK)
    ax.text(.35,-2.0,"circumference × slant length",fontsize=9,color=DARK)
    ax.set_xlim(.2,3.55); ax.set_ylim(-2.35,2.55); ax.axis("off")
    save(fig,"unit04-surface-of-revolution")


def unit04_parametric_motion():
    fig,(ax,)=new_figure(); t=np.linspace(-1.4,1.4,400); x=t+.18*t**3; y=.5+t**2
    ax.plot(x,y,color=BLUE,linewidth=2.6)
    t0=.65; p=np.array([t0+.18*t0**3,.5+t0**2]); v=np.array([1+.54*t0**2,2*t0])
    scale=.55
    arrow(ax,p,p+scale*v,ORANGE,2,13)
    arrow(ax,p,p+np.array([scale*v[0],0]),GREEN,1.8,11)
    arrow(ax,p+np.array([scale*v[0],0]),p+scale*v,GREEN,1.8,11)
    ax.scatter(*p,color=DARK,s=30,zorder=5)
    ax.text(p[0]+.18,p[1]+.55,r"$\mathbf{v}=(dx/dt,dy/dt)$",color=ORANGE,fontsize=10)
    ax.text(p[0]+.24,p[1]-.2,r"$dx/dt$",color=GREEN,fontsize=9)
    ax.text(p[0]+scale*v[0]+.08,p[1]+.2,r"$dy/dt$",color=GREEN,fontsize=9)
    ax.text(-1.75,2.75,r"$\dfrac{ds}{dt}=\sqrt{(dx/dt)^2+(dy/dt)^2}$",fontsize=12,color=DARK)
    style_axis(ax); ax.set_xlim(-1.9,2.15); ax.set_ylim(.2,3.2)
    save(fig,"unit04-parametric-motion")


def unit04_polar_area_element():
    fig,(ax,)=new_figure(); theta=.55; dtheta=.22; r=2.55
    ax.add_patch(Wedge((0,0),r,np.degrees(theta),np.degrees(theta+dtheta),facecolor=PALE_GREEN,
                       edgecolor=GREEN,linewidth=1.8))
    for angle in (theta,theta+dtheta):
        ax.plot([0,r*np.cos(angle)],[0,r*np.sin(angle)],color=BLUE,linewidth=2)
    arc=Arc((0,0),1.5,1.5,theta1=np.degrees(theta),theta2=np.degrees(theta+dtheta),color=ORANGE,linewidth=1.6)
    ax.add_patch(arc)
    ax.text(.72*np.cos(theta+.11),.72*np.sin(theta+.11)+.08,r"$d\theta$",color=ORANGE,fontsize=10)
    ax.text(1.18*np.cos(theta)-.08,1.18*np.sin(theta)-.18,r"$r$",color=BLUE,fontsize=11)
    ax.text(-.2,2.85,r"$dA\approx\frac{1}{2}r^2\,d\theta$",fontsize=14,color=DARK)
    ax.text(-.2,-.55,r"$A=\frac{1}{2}\int_{\alpha}^{\beta}r(\theta)^2\,d\theta$",fontsize=12,color=DARK)
    ax.set_aspect("equal"); ax.set_xlim(-.45,3.25); ax.set_ylim(-.75,3.3); ax.axis("off")
    save(fig,"unit04-polar-area-element")


def unit05_growth_hierarchy():
    fig,(ax,)=new_figure(); x=np.linspace(2,10,500)
    functions=[(np.log(x),r"$\ln x$",DARK),(np.sqrt(x),r"$\sqrt{x}$",GREEN),(x,r"$x$",BLUE),
               (x**2,r"$x^2$",ORANGE),(np.exp(x),r"$e^x$",RED)]
    assert np.log(10)<np.sqrt(10)<10<100<np.exp(10)
    for values,label,color in functions: ax.plot(x,values,label=label,color=color,linewidth=2.2)
    ax.set_yscale("log"); style_axis(ax,y_label="value\n(log scale)")
    ax.set_xlim(2,10); ax.set_ylim(.5,4e4)
    ax.text(2.25,1.5e4,r"$\ln x\ll x^p\ll a^x$ as $x\to\infty$",fontsize=11,color=DARK)
    ax.legend(frameon=False,fontsize=9,ncol=5,loc="lower right")
    save(fig,"unit05-growth-hierarchy")


def unit05_improper_integrals():
    fig,axes=new_figure(2); x=np.linspace(1,9,600)
    for ax,power,title,color in [(axes[0],2,r"$\int_1^\infty x^{-2}dx=1$",GREEN),
                                 (axes[1],1,r"$\int_1^\infty x^{-1}dx=\infty$",RED)]:
        y=x**(-power); ax.plot(x,y,color=BLUE,linewidth=2.5)
        ax.fill_between(x,0,y,color=PALE_GREEN if power==2 else "#F8DADA",alpha=.9)
        ax.text(1.25,.88,title,fontsize=11,color=color)
        ax.text(5.2,.35,"tail continues",fontsize=8,color=DARK)
        arrow(ax,(6.2,.29),(8.4,.18),DARK,1,9)
        style_axis(ax); ax.set_xlim(1,9); ax.set_ylim(0,1.1)
    save(fig,"unit05-improper-integrals")


def unit05_series_partial_sums():
    fig,(ax,)=new_figure(); n=np.arange(1,81)
    geometric=np.cumsum(.5**(n-1)); harmonic=np.cumsum(1/n)
    assert close_enough(geometric[-1],2,1e-12) and harmonic[-1]>harmonic[39]
    ax.plot(n,geometric,color=GREEN,linewidth=2.5,label=r"$\sum_{k=1}^n 2^{1-k}\to2$")
    ax.plot(n,harmonic,color=ORANGE,linewidth=2.5,label=r"$\sum_{k=1}^n 1/k$ grows without bound")
    ax.axhline(2,color=GREEN,linestyle="--",linewidth=1.1)
    ax.text(43,2.08,"limit = 2",color=GREEN,fontsize=9)
    ax.text(8,4.55,"Divergence can be very slow.",color=DARK,fontsize=10)
    style_axis(ax,"n",r"$S_n$"); ax.set_xlim(1,80); ax.set_ylim(.8,5.1)
    ax.legend(frameon=False,fontsize=9,loc="center right")
    save(fig,"unit05-series-partial-sums")


def unit05_taylor_error():
    fig,axes=new_figure(2); ax,bx=axes; x=np.linspace(-1.7,1.7,600); f=np.exp(x)
    polynomials={1:1+x,2:1+x+x**2/2,4:1+x+x**2/2+x**3/6+x**4/24}
    ax.plot(x,f,color=BLUE,linewidth=2.7,label=r"$e^x$")
    for degree,color in [(1,ORANGE),(2,GREEN),(4,RED)]:
        ax.plot(x,polynomials[degree],color=color,linewidth=1.8,label=rf"$P_{degree}$")
    ax.scatter([0],[1],color=DARK,s=30,zorder=5); style_axis(ax); ax.set_xlim(-1.7,1.7); ax.set_ylim(-.2,5.8)
    ax.legend(frameon=False,fontsize=8,ncol=2)
    errors={degree:np.abs(f-values) for degree,values in polynomials.items()}
    assert errors[4][np.argmin(abs(x-1))] < errors[2][np.argmin(abs(x-1))] < errors[1][np.argmin(abs(x-1))]
    for degree,color in [(1,ORANGE),(2,GREEN),(4,RED)]:
        bx.semilogy(x,np.maximum(errors[degree],1e-9),color=color,linewidth=2,label=rf"$|e^x-P_{degree}(x)|$")
    bx.axvspan(-.5,.5,color=PALE_BLUE,alpha=.45); style_axis(bx,y_label="absolute error\n(log scale)")
    bx.set_xlim(-1.7,1.7); bx.set_ylim(1e-8,10); bx.legend(frameon=False,fontsize=8,loc="lower right")
    save(fig,"unit05-taylor-error")


FIGURES = [
    unit01_secant_tangent, unit01_trig_squeeze, unit01_inverse_reflection, unit01_exp_log,
    unit01_hyperbolic, unit02_linear_quadratic, unit02_curve_sign_chart,
    unit02_related_rates_cone, unit02_newton_method, unit02_mean_value_theorem,
    unit02_separation_of_variables, unit03_riemann_sums, unit03_ftc_accumulator,
    unit03_signed_area, unit03_volume_methods, unit03_numerical_integration,
    unit04_trig_substitution_triangles, unit04_arc_length_element,
    unit04_surface_of_revolution, unit04_parametric_motion, unit04_polar_area_element,
    unit05_growth_hierarchy, unit05_improper_integrals, unit05_series_partial_sums,
    unit05_taylor_error,
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "axes.labelsize": 10, "axes.titlesize": 12, "text.color": DARK,
        "figure.constrained_layout.use": True,
    })
    for draw in FIGURES:
        draw()
    generated = sorted(OUTPUT.glob("unit0[1-5]-*.png"))
    if len(generated) != len(FIGURES):
        raise RuntimeError(f"Expected {len(FIGURES)} PNG files, found {len(generated)}")
    print(f"Generated {len(generated)} figures in {OUTPUT}")


if __name__ == "__main__":
    main()
