#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 LLM 对话循环的"交流模式"漂移。

指标（逐条计算，再按时间窗口聚合）
  长度        每条消息字符数
  问句率      含 ？/? 的比例
  第二人称    你/您 密度（每千字）—— 直接称呼对方，掉下去说明变成独白
  元话语      我们/对话/这场/话题 密度 —— 开始谈论"这场对话本身"
  一致/分歧   认同词 vs 转折词的比例 —— 讨好式附和螺旋的代理指标
  接话度      与上一条消息的字符 3-gram Jaccard（对比随机远距离配对的基线）
  新颖度      1 - 与之前所有消息的最大 TF-IDF 余弦相似度（掉到 0 = 不再说新东西）
  窗口多样性  连续 20 条内部的两两余弦距离均值
  离题度      与开场话题的 TF-IDF 余弦相似度
  自重复      与最近 20 条的最大 3-gram Jaccard

用法
  python3 analyze.py runs/loop-20261013-1500
  python3 analyze.py runs/loop-* runs/solo-*      # 多条件对比
"""

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcache-"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

WIN = 20          # 滑动窗口宽度（条）
WINDOW_LEN = WIN

MARKERS = {
    "addr": re.compile(r"[你您]"),
    "meta": re.compile(r"我们|对话|这场|这段|聊天|交流|这个话题|循环"),
    "agree": re.compile(r"你说得对|说得对|是的|对的|没错|确实|同意|认同|正是|有道理"),
    "disagree": re.compile(r"但是|不过|然而|未必|不一定|不同意|并不|问题在于|反过来|其实不"),
    "hedge": re.compile(r"也许|可能|大概|似乎|某种程度上|某种意义上"),
    "question": re.compile(r"[？?]"),
    "struct": re.compile(r"(?m)^\s*(?:[-*#]|\d+[.、])|\*\*|^\s*[^\n：:]{2,12}：\s*$"),
}


def grams(text: str, n=3) -> set:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rolling(x, w=WIN):
    """居中滑动均值；NaN 自动跳过（窗口内全是 NaN 才返回 NaN）。"""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(w, center=True, min_periods=max(2, w // 4)).mean().to_numpy()


def load_run(run_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    if recs and not meta.get("config"):   # 跑到一半被杀：从记录里凑出配置和合计
        meta = {"config": {"model": recs[0].get("model"), "condition": recs[0].get("condition"),
                           "temperature": "?", "topic": ""},
                "totals": {"prompt_tokens": sum(r["prompt_tokens"] for r in recs),
                           "cache_hit": sum(r["cache_hit"] for r in recs),
                           "cache_miss": sum(r["cache_miss"] for r in recs),
                           "completion_tokens": sum(r["completion_tokens"] for r in recs),
                           "usd": round(sum(r["cost_usd"] for r in recs), 4)},
                "partial": True}
    return recs, meta


def metrics(recs: list, topic: str) -> list:
    texts = [r["content"] for r in recs]
    n = len(texts)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    M = vec.fit_transform(texts + [topic])
    S = (M @ M.T).toarray()          # 已 L2 归一化 -> 余弦相似度
    seed = S[-1, :-1]

    gs = [grams(t) for t in texts]
    gap = max(5, n // 10)            # "远距离配对"基线：短对话也适用
    out = []
    for i, (r, t) in enumerate(zip(recs, texts)):
        L = max(1, len(t))
        prev = jaccard(gs[i], gs[i - 1]) if i > 0 else np.nan
        base = [jaccard(gs[i], gs[j]) for j in range(n) if abs(j - i) >= gap]
        selfrep = max([jaccard(gs[i], gs[j]) for j in range(max(0, i - 20), i)], default=np.nan)
        novelty = 1 - S[i, :i].max() if i > 0 else np.nan
        out.append({
            "i": i, "round": r["round"], "agent": r["agent"], "chars": L,
            "tokens_out": r.get("completion_tokens"),
            "addr": len(MARKERS["addr"].findall(t)) / L * 1000,
            "meta": len(MARKERS["meta"].findall(t)) / L * 1000,
            "hedge": len(MARKERS["hedge"].findall(t)) / L * 1000,
            "question": int(bool(MARKERS["question"].search(t))),
            "struct": len(MARKERS["struct"].findall(t)) / L * 1000,
            "agree": len(MARKERS["agree"].findall(t)),
            "disagree": len(MARKERS["disagree"].findall(t)),
            "overlap_prev": prev,
            "overlap_base": float(np.median(base)) if base else np.nan,
            "engage_ratio": (prev / np.median(base)) if base and np.median(base) > 0 else np.nan,
            "selfrep": selfrep,
            "novelty": novelty,
            "seed_sim": float(seed[i]),
            "q_content": (len(MARKERS["agree"].findall(t)) + len(MARKERS["disagree"].findall(t))) / L * 1000,
        })

    div = np.full(n, np.nan)
    homog = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - WIN // 2), min(n, i + WIN // 2 + 1)
        blk = S[lo:hi, lo:hi]
        iu = np.triu_indices(len(blk), 1)
        if len(iu[0]):
            div[i] = 1 - blk[iu].mean()
        # 甲乙同质化：与对方消息的平均相似度 ÷ 与同一个人其它消息的平均相似度
        idx = list(range(lo, hi))
        cross = [S[i, j] for j in idx if j != i and recs[j]["agent"] != recs[i]["agent"]]
        same = [S[i, j] for j in idx if j != i and recs[j]["agent"] == recs[i]["agent"]]
        if cross and same and np.mean(same) > 0:
            homog[i] = float(np.mean(cross) / np.mean(same))
    for i in range(n):
        out[i]["window_diversity"] = float(div[i])
        out[i]["homog"] = float(homog[i])
    return out


def summarize(m: list, label: str) -> dict:
    """分四段（Q1..Q4）看漂移。"""
    n = len(m)
    qs = np.array_split(np.arange(n), 4)
    keys = ["chars", "addr", "meta", "question", "overlap_prev", "overlap_base", "engage_ratio",
            "selfrep", "novelty", "seed_sim", "window_diversity", "struct", "tokens_out"]

    def agg(idx, k):
        v = [m[i][k] for i in idx if m[i].get(k) is not None and not (isinstance(m[i][k], float) and np.isnan(m[i][k]))]
        return float(np.mean(v)) if v else float("nan")

    rows = {k: [agg(q, k) for q in qs] for k in keys}
    # 认同率 = 认同词 / (认同词+转折词)
    for qi, q in enumerate(qs):
        a = sum(m[i]["agree"] for i in q)
        d = sum(m[i]["disagree"] for i in q)
        rows.setdefault("agree_ratio", []).append(a / (a + d) if a + d else float("nan"))
    return {"label": label, "n": n, "quarters": rows}


def setup_cjk_font():
    from matplotlib import font_manager, rcParams
    for p in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Light.ttc"]:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                rcParams["font.family"] = name
                return True
            except Exception:
                continue
    return False


def make_figure(runs, metrics_by_run, out_png: Path, cjk: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)
    panels = [
        ("chars", L("消息长度（字符）", "message length (chars)"), False),
        ("engage_ratio", L("接话度：与上一条重叠 / 随机基线", "engagement: overlap w/ prev / baseline"), False),
        ("agree_ratio_roll", L("认同词占比（滚动）", "agreement marker share (roll)"), False),
        ("novelty", L("新颖度：1 - 与既往最大相似", "novelty: 1 - max sim to earlier"), False),
        ("window_diversity", L("窗口内语义多样性（20 条）", "within-window semantic diversity"), False),
        ("homog", L("甲乙同质化：跨人相似 ÷ 同人相似", "A-B homogenization: cross / within"), False),
        ("addr", L("第二人称密度（每千字）", "2nd-person density (per 1k chars)"), False),
        ("meta", L("元话语密度（每千字）", "meta-discourse density (per 1k chars)"), False),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for ax, (key, title, _) in zip(axes.ravel(), panels):
        for (label, m) in metrics_by_run:
            if key == "agree_ratio_roll":
                a = rolling([x["agree"] for x in m], 10)
                d = rolling([x["disagree"] for x in m], 10)
                y = a / np.maximum(a + d, 1e-9)
            else:
                y = rolling([x.get(key, np.nan) for x in m])
            x = np.linspace(0, 1, len(y))
            ax.plot(x, y, lw=1.8, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(L("对话进度", "progress"))
        if key == "chars":
            ax.set_yscale("log")          # 长度从 400 字塌到 3 字，线性轴看不出来
        ax.grid(alpha=.25)
    axes.ravel()[0].legend(fontsize=9)
    n_msg = max(len(m) for _, m in metrics_by_run)
    fig.suptitle(L(f"两个 LLM 的对话（{n_msg} 条消息）：交流模式漂移",
                   f"Two-LLM dialogue ({n_msg} messages): interaction drift"), fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    runs, all_m = [], []
    for d in args.runs:
        p = Path(d)
        recs, meta = load_run(p)
        topic = (meta.get("config", {}) or {}).get("topic", "")
        cond = (meta.get("config", {}) or {}).get("condition", p.name)
        m = metrics(recs, topic)
        label = f"{p.name} · {cond} · {len(recs)}条"
        runs.append((label, p, recs, meta, m))
        all_m.append((label, m))

        import csv
        with (p / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(m[0].keys()))
            w.writeheader()
            w.writerows(m)

    lines = ["# 交流模式漂移报告", "",
             f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    key_names = {"chars": "长度", "addr": "第二人称/千字", "meta": "元话语/千字", "question": "问句率",
                 "overlap_prev": "接话重叠", "overlap_base": "随机基线", "engage_ratio": "接话度",
                 "selfrep": "自重复", "novelty": "新颖度", "seed_sim": "离题度(与话题相似)",
                 "window_diversity": "窗口多样性", "agree_ratio": "认同词占比", "struct": "结构化密度",
                 "homog": "甲乙同质化", "tokens_out": "输出token"}
    for label, p, recs, meta, m in runs:
        s = summarize(m, label)
        tot = meta.get("totals", {})
        lines += [f"## {label}", "",
                  f"- 目录：`{p}`",
                  f"- 模型/条件：`{meta.get('config',{}).get('model','?')}` / `{meta.get('config',{}).get('condition','?')}`"
                  f"，温度 {meta.get('config',{}).get('temperature','?')}",
                  f"- token：in {tot.get('prompt_tokens',0):,}（缓存命中 {tot.get('cache_hit',0):,}）"
                  f" / out {tot.get('completion_tokens',0):,}，花费 ${tot.get('usd',0)}", "",
                  "| 指标 | Q1 | Q2 | Q3 | Q4 | Q4-Q1 |", "|---|---|---|---|---|---|"]
        for k, name in key_names.items():
            if k not in s["quarters"]:
                continue
            v = s["quarters"][k]
            lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" if abs(x) < 100 else f"{x:.0f}" for x in v)
                         + f" | {v[3]-v[0]:+.3f} |")
        lines.append("")
    (Path(args.runs[0]).parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    cjk = setup_cjk_font()
    png = make_figure(runs, all_m, Path(args.runs[0]).parent / "drift.png", cjk)
    print("\n".join(lines))
    print(f"\n图：{png}\nCSV：<run>/metrics.csv\n报告：{Path(args.runs[0]).parent/'summary.md'}")


if __name__ == "__main__":
    main()
