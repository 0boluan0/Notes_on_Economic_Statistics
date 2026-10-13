#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可读性测量：离线指标 + LLM 评审。

离线指标（不用 API，完全透明可复核）
  平均句长/最长句   按 。！？； 切分，中文长句是"读不懂"的第一来源
  比喻密度          "像…一样 / 好比 / 仿佛 / 如同 / 宛如 / 似的 / 就像" 每千字
  具体性密度        时间、人物、地点、数字、器物等可想象成分 每千字
  累计自造说法      引号里出现过的、之前没出现过的短语总数（读者要同时记住几个）

LLM 评审（--judge N：每个 run 抽 N 条，用 deepseek-v4-pro 打 1–5 分）
  注意：评审模型与生成模型同族，存在同源偏好；分数只用于**同一批 run 之间的相对比较**。

用法
  python3 readability.py runs/loop-a runs/plain-1 runs/task-1
  python3 readability.py runs/* --judge 15
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

BASE_URL = "https://api.deepseek.com"
PRICES = {"deepseek-v4-pro": {"hit": (0.022, 0.044), "miss": (0.66, 1.32), "out": (1.98, 3.96)},
          "deepseek-flash": {"hit": (0.003, 0.006), "miss": (0.15, 0.30), "out": (0.6, 1.2)}}

SIMILE = re.compile(r"像[^，。；]{0,12}一样|好比|仿佛|如同|宛如|似的|就像")
CONCRETE = re.compile(
    r"\d|昨天|今天|前天|明天|早上|晚上|中午|半夜|那年|小时候|我妈|我爸|我奶奶|朋友|同事|"
    r"桌上|桌上|街上|手里|门口|楼下|块钱|分钟|小时|电话|微信|超市|地铁|公交|饭|杯子|灯|钥匙|医院|学校|公司")
QUOTED = re.compile(r'[“"「]([^”"」]{2,12})[”"」]')

JUDGE_SYS = """你是中文可读性评审。面向一个普通中文读者（不是哲学系学生），判断这段话"读一遍能不能看懂"。
只输出 JSON，不要任何别的字：
{"score": 1到5的整数, "jargon": 本段自造的比喻或术语个数（整数）, "why": "不超过12个字的理由"}
评分标准：
5 = 完全看得懂，像正常人在说话
4 = 基本看得懂，个别地方要回读一遍
3 = 大意能抓住，但抽象、费劲
2 = 读一遍抓不住在说什么，要靠猜
1 = 看不懂，像黑话或者谜语"""


def load(d):
    return [json.loads(l) for l in (Path(d) / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def offline(recs):
    cum, seen = [], set()
    per = []
    for r in recs:
        t = r["content"]
        L = max(1, len(t))
        sents = [s for s in re.split(r"[。！？；]", t) if s.strip()]
        new = [q for q in QUOTED.findall(t) if q not in seen]
        seen.update(new)
        cum.append(len(seen))
        per.append({
            "i": r["i"], "chars": L,
            "sent_len": (sum(len(s) for s in sents) / len(sents)) if sents else L,
            "max_sent": max((len(s) for s in sents), default=L),
            "simile": len(SIMILE.findall(t)) / L * 1000,
            "concrete": len(CONCRETE.findall(t)) / L * 1000,
            "jargon": len(seen),
        })
    return per, cum


class Judge:
    def __init__(self, model="deepseek-v4-pro"):
        import requests
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit("缺少 DEEPSEEK_API_KEY")
        self.model = model
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.usd = 0.0
        self.calls = 0

    def __call__(self, text):
        payload = {"model": self.model, "stream": False, "max_tokens": 120, "temperature": 0,
                   "thinking": {"type": "disabled"},
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": text}]}
        for a in range(4):
            try:
                r = self.s.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                j = r.json()
                u = j.get("usage", {})
                p = PRICES[self.model]
                self.usd += (u.get("prompt_cache_hit_tokens", 0) * p["hit"][1]
                             + u.get("prompt_cache_miss_tokens", 0) * p["miss"][1]
                             + u.get("completion_tokens", 0) * p["out"][1]) / 1e6
                self.calls += 1
                txt = j["choices"][0]["message"]["content"].strip().strip("`")
                if txt.startswith("json"):
                    txt = txt[4:]
                return json.loads(txt)
            except Exception as e:
                if a == 3:
                    print(f"    [judge 失败] {e}", flush=True)
                    return None
                time.sleep(2 ** a)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--judge", type=int, default=0, help="每个 run 抽多少条送 LLM 评审")
    ap.add_argument("--judge-model", default="deepseek-v4-pro")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    judge = Judge(a.judge_model) if a.judge else None
    rows, detail = [], {}
    for d in a.runs:
        recs = load(d)
        meta = {}
        mp = Path(d) / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        variant = (meta.get("config", {}) or {}).get("variant", "base")
        per, cum = offline(recs)
        early = per[:20]
        row = {
            "run": Path(d).name, "variant": variant, "msgs": len(recs),
            "平均句长": sum(x["sent_len"] for x in early) / len(early),
            "最长句": max(x["max_sent"] for x in early),
            "比喻/千字": sum(x["simile"] for x in early) / len(early),
            "具体性/千字": sum(x["concrete"] for x in early) / len(early),
            "前14条累计自造说法": cum[min(13, len(cum) - 1)],
        }
        if judge:
            rnd = random.Random(a.seed)
            k = min(a.judge, len(recs))
            idxs = sorted(set(list(range(0, min(12, len(recs)))) + rnd.sample(range(min(12, len(recs)), len(recs)), max(0, k - min(12, len(recs))))))
            sc, jar = [], []
            for i in idxs:
                res = judge(recs[i]["content"])
                if res and isinstance(res.get("score"), (int, float)):
                    sc.append(float(res["score"]))
                    if isinstance(res.get("jargon"), (int, float)):
                        jar.append(float(res["jargon"]))
            row["可读性均分(1-5)"] = sum(sc) / len(sc) if sc else float("nan")
            row["自造比喻/条"] = sum(jar) / len(jar) if jar else float("nan")
            row["评审条数"] = len(sc)
            detail[Path(d).name] = {"scores": sc, "idxs": idxs}
        rows.append(row)

    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(f"{r[k]:.1f}" if isinstance(r[k], float) else str(r[k])) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(widths[k]) for k in keys))
    print(" | ".join("-" * widths[k] for k in keys))
    for r in rows:
        print(" | ".join((f"{r[k]:.1f}" if isinstance(r[k], float) else str(r[k])).ljust(widths[k]) for k in keys))
    if judge:
        print(f"\n评审调用 {judge.calls} 次，花费 ${judge.usd:.4f}（模型 {a.judge_model}）")
    out = Path(a.runs[0]).parent / "readability.json"
    out.write_text(json.dumps({"rows": rows, "detail": detail}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细：{out}")


if __name__ == "__main__":
    main()
