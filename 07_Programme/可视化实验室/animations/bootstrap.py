"""Bootstrap film. Render from the laboratory's seeded numerical model.

manim -qh --media_dir /tmp/visual-lab-manim animations/bootstrap.py Bootstrap
Requires Manim 0.21.0 with typst; no TeX installation or model API.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from manim import *

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTED = "#111820", "#F4F2ED", "#9AAAB9"
BLUE, GREEN, GOLD, RULE = "#88BCF5", "#85D7BE", "#F0C47C", "#344454"
PARAMS = dict(n=8, reps=500, distribution="exponential", seed=42)
payload = json.loads(subprocess.check_output([
    "node", "-e",
    "const M=require('./assets/math.js');"
    "console.log(JSON.stringify(M.compute('bootstrap'," + json.dumps(PARAMS) + ")))"
], cwd=ROOT, text=True))
D = payload["data"]
SAMPLE = np.array(D["sample"])
MEANS = np.array(D["bootstrapMeans"])
ROWS = D["resampleExamples"]
LETTERS = "ABCDEFGH"


def check_data():
    """Independent arithmetic checks, also run before every render."""
    assert payload["params"] == PARAMS
    assert len(MEANS) == 500 and len(SAMPLE) == 8
    for i, row in enumerate(ROWS):
        assert len(row) == 8 and all(0 <= j < 8 for j in row)
        assert abs(SAMPLE[row].mean() - MEANS[i]) < 1e-12
    assert np.allclose(np.quantile(MEANS, [.025, .975]), D["interval"], rtol=0, atol=1e-12)
    assert abs(SAMPLE.std(ddof=0) / np.sqrt(8) - D["conditionalExactSE"]) < 1e-12
    assert sum(D["histogram"]["counts"]) == len(MEANS)
    assert [ROWS[1].count(i) for i in range(8)] == [0, 2, 1, 0, 0, 2, 3, 0]
    print("PASS: six actual resamples, means, quantiles, conditional SE and 500 histogram points")


check_data()


def label(s, size=28, color=INK, **kwargs):
    return Text(s, font="PingFang SC", font_size=size, color=color, **kwargs)


def english(s, size=18, color=MUTED):
    return Text(s, font="Avenir Next", font_size=size, color=color)


def formula(s, size=38, color=INK):
    return MathTypst(s, font_size=size, color=color)


def token(index, position, color=BLUE, scale=1):
    ring = Circle(radius=.31, color=color, stroke_width=2).set_fill(BG, opacity=1)
    identity = english(LETTERS[index], 26, color).move_to(ring)
    value = english(f"{SAMPLE[index]:.3f}", 23, INK).next_to(ring, DOWN, buff=.15)
    return VGroup(ring, identity, value).scale(scale).move_to(position)


class Bootstrap(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.marks = {}
        self.heading = VGroup()
        self.caption = VGroup()

        def title(number, text, subtitle):
            new = VGroup(
                english(f"BOOTSTRAP   /   {number:02d}", 17, GREEN).move_to([-5.4, 3.46, 0], aligned_edge=LEFT),
                label(text, 42).move_to([-5.4, 2.91, 0], aligned_edge=LEFT),
                english(subtitle, 18).move_to([-5.4, 2.43, 0], aligned_edge=LEFT),
            )
            if len(self.heading):
                self.play(FadeOut(self.heading), FadeIn(new, shift=.12 * UP), run_time=.65)
            else:
                self.add(new)
            self.heading = new

        def caption(zh, en):
            new = VGroup(label(zh, 27), english(en, 18)).arrange(DOWN, buff=.17).move_to([0, -3.07, 0])
            if new.width > 12.2:
                new.scale_to_fit_width(12.2)
            if len(self.caption):
                self.play(FadeOut(self.caption), FadeIn(new), run_time=.35)
            else:
                self.add(new)
            self.caption = new

        def mark(name):
            self.marks[name] = round(self.time, 3)

        title(1, "只有一份样本，怎样看见不确定性？", "One observed sample. An uncertain population mean.")
        demo_note = label("合成例子 / synthetic data · 数值保留三位小数", 18, MUTED).move_to([0, 1.89, 0])
        self.add(demo_note)
        xs = np.linspace(-4.7, 4.7, 8)
        base = -.85
        bars = VGroup()
        for i, x in enumerate(xs):
            h = SAMPLE[i] * 1.15
            bar = Rectangle(width=.43, height=h, fill_color=BLUE, fill_opacity=.82, stroke_width=0)
            bar.move_to([x, base + h / 2, 0])
            value = english(f"{SAMPLE[i]:.3f}", 23, BLUE).next_to(bar, UP, buff=.15)
            letter = english(LETTERS[i], 23).move_to([x, base - .3, 0])
            bars.add(VGroup(bar, value, letter))
        baseline = Line([-5.3, base, 0], [5.3, base, 0], color=RULE, stroke_width=1.6)
        # A complete first frame gives Obsidian a useful native video poster.
        self.add(baseline, bars)
        mean = formula(rf"n = 8 quad overline(x) = 1/8 sum_(i=1)^8 x_i approx {D['sampleMean']:.3f}", 39).move_to([0, -1.88, 0])
        self.add(mean)
        caption("均值是一个数；我们还想知道，它有多不稳定。", "A point estimate is one number. How much might it vary?")
        mark("observed_sample")
        self.wait(6)

        title(2, "把这八个观测，当作抽样的来源", "Resample the observed values, with replacement.")
        sources = VGroup(*[token(i, [xs[i], 1.12, 0]) for i in range(8)])
        self.play(ReplacementTransform(bars, sources), FadeOut(baseline), FadeOut(mean), FadeOut(demo_note), run_time=1.1)
        source_name = label("原始样本 · 保持不变", 20, MUTED).move_to([-5.45, 1.9, 0], aligned_edge=LEFT)
        sample_name = label("一份重抽样 · 抽满 8 次", 20, GREEN).move_to([-5.45, -.15, 0], aligned_edge=LEFT)
        slots = VGroup(*[Circle(radius=.31, color=RULE, stroke_width=1.5).move_to([x, -.91, 0]) for x in xs])
        self.play(FadeIn(source_name), FadeIn(sample_name), FadeIn(slots), run_time=.5)
        caption("每抽一次，原来的八项仍然都可被抽到。", "Every draw independently selects one of the same eight observations.")
        draws = VGroup()
        for j, index in enumerate(ROWS[1]):
            drawn = token(index, [xs[j], -1.12, 0], GREEN)
            # Match each circle center to its empty slot, not the group's center.
            drawn.shift(slots[j].get_center() - drawn[0].get_center())
            self.play(
                Indicate(sources[index][0], color=GOLD, scale_factor=1.14),
                TransformFromCopy(sources[index], drawn, path_arc=.18),
                FadeOut(slots[j]), run_time=.7,
            )
            draws.add(drawn)
            if j == 4:
                repeat = SurroundingRectangle(VGroup(draws[3], draws[4]), color=GOLD, buff=.14, corner_radius=.12)
                caption("G 连续出现两次，因为抽取后仍可再抽。", "G appears twice in a row: replacement makes repeated selection possible.")
                self.play(Create(repeat), run_time=.4)
                self.wait(1.9)
                self.play(FadeOut(repeat), run_time=.3)
        caption("抽了八次，只有四种身份：B、C、F、G。", "Eight draws; four distinct observations. The multiplicities matter.")
        mark("replacement")
        self.wait(2.5)
        equation = formula(rf"overline(x)^* = (2x_B + x_C + 2x_F + 3x_G) / 8 approx {MEANS[1]:.3f}", 35, GOLD).move_to([0, -2.14, 0])
        self.play(FadeIn(equation), run_time=.9)
        mark("resample_mean")
        self.wait(3.0)

        title(3, "每份重抽样，只留下一个均值", "One resample becomes one point.")
        self.play(FadeOut(sources), FadeOut(source_name), FadeOut(sample_name), run_time=.5)
        line_y = -1.18
        xcoord = lambda v: -5.05 + (v - .25) / 1.1 * 10.1
        axis = Line([xcoord(.25), line_y, 0], [xcoord(1.35), line_y, 0], color=RULE, stroke_width=2)
        ticks = VGroup()
        for v in [.4, .6, .8, 1.0, 1.2]:
            ticks.add(Line([xcoord(v), line_y, 0], [xcoord(v), line_y - .08, 0], color=MUTED, stroke_width=1.5))
            ticks.add(english(f"{v:.1f}", 21).move_to([xcoord(v), line_y - .35, 0]))
        axis_name = label("重抽样均值", 19, MUTED).move_to([4.35, -.65, 0])
        self.play(draws.animate.scale(.77).move_to([0, .83, 0]), equation.animate.scale(.8).move_to([0, -.12, 0]), Create(axis), FadeIn(ticks), FadeIn(axis_name), run_time=1)
        dot2 = Dot([xcoord(MEANS[1]), line_y + .07, 0], radius=.065, color=GOLD)
        self.play(ReplacementTransform(draws, dot2), FadeOut(equation), run_time=1.25)
        caption("图上的一个点，代表整整一份样本的均值。", "A point represents the mean of eight draws, not one observation.")
        points = {1: dot2}
        result_label = formula(rf"overline(x)^* approx {MEANS[1]:.3f}", 42, GOLD).move_to([0, .68, 0])
        self.play(FadeIn(result_label), run_time=.45)
        mark("one_mean_one_point")
        self.wait(2.7)

        for i in [0, 2, 3, 4, 5]:
            mini = VGroup(*[token(j, [xs[k] * .63, .7, 0], GREEN, .66) for k, j in enumerate(ROWS[i])])
            current_label = formula(rf"overline(x)^* approx {MEANS[i]:.3f}", 31, GOLD).move_to([0, -.22, 0])
            self.play(FadeOut(result_label), FadeIn(mini), FadeIn(current_label), run_time=.45)
            self.wait(.4)
            point = Dot([xcoord(MEANS[i]), line_y + .07, 0], radius=.065, color=GOLD)
            self.play(ReplacementTransform(mini, point), run_time=.65)
            points[i] = point
            result_label = current_label
        self.play(FadeOut(result_label), run_time=.3)
        mark("six_means")
        self.wait(1.5)

        title(4, "让重复的结果，形成一幅分布", "The spread of the resample means estimates uncertainty.")
        caption("按数值区间分组；每个金色小点仍是一份重抽样。", "Group means into bins. Each gold dot still stands for one resample.")
        self.play(axis.animate.shift(.63 * DOWN), ticks.animate.shift(.63 * DOWN), axis_name.animate.shift(.63 * DOWN), run_time=.6)
        hist_base = line_y - .63
        h = D["histogram"]
        counts = [0] * len(h["counts"])
        destinations = []
        for value in MEANS:
            b = min(len(counts) - 1, int((value - h["domain"][0]) / h["width"]))
            counts[b] += 1
            destinations.append([xcoord(h["points"][b][0]), hist_base + .077 * (counts[b] - .5), 0])
        assert counts == h["counts"]
        all_dots = VGroup(*[Dot(p, radius=.030, color=GOLD) for p in destinations])
        self.play(*[Transform(points[i], all_dots[i]) for i in range(6)], run_time=.65)
        counter = english("6 resamples", 21, GOLD).move_to([4.13, 1.7, 0])
        self.play(FadeIn(counter), run_time=.3)
        for start, end, duration in [(6, 25, 1.1), (25, 100, 1.6), (100, 250, 1.8), (250, 500, 2.1)]:
            new_counter = english(f"{end} resamples", 21, GOLD).move_to(counter)
            self.play(LaggedStart(*[FadeIn(all_dots[i], shift=.2 * DOWN) for i in range(start, end)], lag_ratio=.025), Transform(counter, new_counter), run_time=duration)
        self.remove(*points.values())
        self.add(all_dots)
        mid_x = xcoord(D["sampleMean"])
        center_line = DashedLine([mid_x, hist_base, 0], [mid_x, 1.64, 0], dash_length=.08, color=BLUE, stroke_width=1.8)
        center_label = formula(rf"overline(x) approx {D['sampleMean']:.3f}", 29, BLUE).move_to([mid_x - 1.05, 1.88, 0])
        self.play(Create(center_line), FadeIn(center_label), run_time=.65)
        caption("它围绕当前样本的均值波动，而非自动找回总体真值。", "The bootstrap is centered on this sample's mean, not the unknown truth.")
        mark("bootstrap_distribution")
        self.wait(3.1)

        # Each rectangle has the exact height of the stack it replaces.
        hist_bars = VGroup()
        for j, c in enumerate(counts):
            height = max(.004, .077 * c)
            rect = Rectangle(width=xcoord(h["width"] + .25) - xcoord(.25) - .022,
                             height=height, stroke_width=0, fill_color=GREEN, fill_opacity=.7 if c else 0)
            rect.move_to([xcoord(h["points"][j][0]), hist_base + height / 2, 0])
            hist_bars.add(rect)
        self.play(FadeOut(all_dots), FadeIn(hist_bars), run_time=.9)
        title(5, "取两端的分位数，得到百分位区间", "The 2.5th and 97.5th percentiles form a nominal 95% interval.")
        self.play(FadeOut(center_label), FadeOut(counter), run_time=.4)
        lo, hi = D["interval"]
        interval_line = Line([xcoord(lo), hist_base - .13, 0], [xcoord(hi), hist_base - .13, 0], color=GOLD, stroke_width=6)
        endpoints = VGroup()
        for v, pct, direction in [(lo, "2.5%", LEFT), (hi, "97.5%", RIGHT)]:
            marker = DashedLine([xcoord(v), hist_base, 0], [xcoord(v), 1.15, 0], dash_length=.06, color=GOLD, stroke_width=2)
            tag = VGroup(english(pct, 21, GOLD), english(f"{v:.3f}", 31, GOLD)).arrange(DOWN, buff=.13).move_to([xcoord(v) + direction[0] * .35, 1.67, 0])
            endpoints.add(marker, tag)
        self.play(Create(interval_line), LaggedStart(*[Create(e) for e in endpoints], lag_ratio=.12), run_time=1.2)
        caption("左右各留下 2.5%，两端之间构成一个百分位区间。", "The interval lies between the 2.5th and 97.5th sample quantiles.")
        interval = formula(rf"I_* = [q_(.025)^*, q_(.975)^*] approx [{lo:.3f}, {hi:.3f}]", 42, GOLD).move_to([0, -.05, 0])
        # The equation follows the graph; it never obscures the distribution.
        self.wait(2.5)
        self.play(FadeOut(hist_bars), FadeOut(center_line), FadeIn(interval),
                  Transform(endpoints[0], Line([xcoord(lo), hist_base - .28, 0], [xcoord(lo), hist_base + .05, 0], color=GOLD)),
                  Transform(endpoints[2], Line([xcoord(hi), hist_base - .28, 0], [xcoord(hi), hist_base + .05, 0], color=GOLD)), run_time=.7)
        caption("名义 95% 描述区间方法；小样本并无精确覆盖保证。", "Nominal 95% is not an exact coverage guarantee, especially for a small sample.")
        mark("percentile_interval")
        self.wait(4.5)

        old_graph = VGroup(axis, ticks, axis_name, interval_line, endpoints, interval)
        self.play(FadeOut(old_graph), run_time=.7)
        title(6, "多重抽几次，会多出新信息吗？", "More simulation does not create more observed data.")
        small_sources = VGroup(*[token(i, [xs[i] * .8, 1.07, 0], BLUE, .82) for i in range(8)])
        self.play(FadeIn(small_sources), run_time=.65)
        n_text = formula("n = 8", 56, BLUE).move_to([-2.7, -.73, 0])
        b_text = formula("B = 500", 56, GOLD).move_to([2.1, -.73, 0])
        n_label = label("原始观测", 25, MUTED).next_to(n_text, DOWN, buff=.3)
        b_label = label("模拟重复", 25, MUTED).next_to(b_text, DOWN, buff=.3)
        divider = Line([0, -.25, 0], [0, -1.65, 0], color=RULE, stroke_width=1.5)
        self.play(FadeIn(n_text), FadeIn(b_text), FadeIn(n_label), FadeIn(b_label), Create(divider), run_time=.7)
        caption("增加 B，减少的是模拟误差；原始数据仍只有八项。", "Increasing B reduces Monte Carlo error. The eight observations stay the same.")
        self.play(Transform(b_text, formula("B = 2000", 56, GOLD).move_to(b_text)), run_time=1.7)
        self.wait(2.6)
        caption("没观测到的尾部、偏差或依赖结构，不会靠重复自动补全。", "Resampling cannot recover unseen tails or repair biased or dependent sampling.")
        mark("simulation_vs_information")
        self.wait(3.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.65)
        closing = VGroup(
            label("现在，亲手改变一个条件。", 44),
            english("Keep the sample fixed. Increase B. Then change n.", 23),
            label("在下方实验中比较 B 与 n 的作用", 27, GREEN),
            english("Illustration: n = 8 · seed = 42 · Exponential(1) · 500 resamples", 17),
        ).arrange(DOWN, buff=.38).move_to(ORIGIN)
        self.play(FadeIn(closing, shift=.12 * UP), run_time=.7)
        mark("next_experiment")
        self.wait(3.3)
        self.marks["duration"] = round(self.time, 3)
        Path(config.media_dir, "bootstrap-timeline.json").write_text(json.dumps(self.marks, indent=2) + "\n")


if __name__ == "__main__":
    print(json.dumps({"params": PARAMS, "sample_mean": D["sampleMean"], "interval": D["interval"]}, indent=2))
