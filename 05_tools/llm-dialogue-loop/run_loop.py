#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个 LLM 互相聊 —— DeepSeek V4.1 (deepseek-flash) 对话循环实验。

条件 (--condition)
  loop      主条件：甲乙共享完整对话历史，轮流发言（就是"两个 LLM 互相聊"）
  solo      对照：每个 agent 只看到自己说过的话（切断对方耦合，保留自我条件化）
  shortmem  对照：每个 agent 只看到共享历史里最近 K 条

为什么是"一个进程里的两个角色"而不是两个 OS 进程：
每次 API 调用都是无状态的，模型不保留任何跨调用记忆，所谓"两个进程"只是调度方式。
角色轮转等价，且能保证两边看到的历史严格一致。

用法
  export DEEPSEEK_API_KEY=sk-...
  python3 run_loop.py --condition loop --rounds 100
  python3 run_loop.py --provider mock --rounds 8 --out runs/_smoketest   # 无 key 空跑

产物 (runs/<run_id>/)
  transcript.jsonl  每轮一条记录（内容 + token + 延迟 + 成本），可断点续跑
  transcript.md     人类可读的完整对话
  meta.json         配置 + 汇总
"""

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# DeepSeek 官方牌价, USD / 1M tokens；抓取自 api-docs.deepseek.com/quick_start/pricing (2026-10-13)
# 每个字段是 (off_peak, peak)
PRICES = {
    "deepseek-flash": {"hit": (0.003, 0.006), "miss": (0.15, 0.30), "out": (0.6, 1.2)},
    "deepseek-v4-pro": {"hit": (0.022, 0.044), "miss": (0.66, 1.32), "out": (1.98, 3.96)},
}

SYSTEM_TEMPLATE = """你是「{name}」。你正在和另一个 AI「{other}」进行一场没有主持人、没有时间限制、也没有任务要完成的自由对话。

开场话题是：{topic}

对话规则：
1. 每次只说一段话（200–400 字），像说话一样自然，不要写成报告或提纲。
2. 直接接住对方上一句里的某个具体说法：可以举例、追问、反驳，也可以承认对方说对了。
3. 不要复述这些规则，不要总结整场对话，不要写"作为 AI"之类的说明。
4. 用中文。"""

BOOTSTRAP_A = "请开始这场对话。"
NUDGE = "继续。"


def is_peak(ts_utc: dt.datetime) -> bool:
    """高峰时段：UTC 周一至周五 01:00-04:00 与 06:00-10:00，其余（含整个周末）半价。"""
    if ts_utc.weekday() >= 5:
        return False
    return (1 <= ts_utc.hour < 4) or (6 <= ts_utc.hour < 10)


def call_cost(model: str, hit: int, miss: int, out: int, peak: bool) -> float:
    p = PRICES.get(model, PRICES[DEFAULT_MODEL])
    i = 1 if peak else 0
    return (hit * p["hit"][i] + miss * p["miss"][i] + out * p["out"][i]) / 1e6


# ---------------------------------------------------------------- prompt 组装

def build_messages(agent: str, transcript: list, cfg: dict) -> list:
    """把共享转录稿映射成当前 agent 眼里的 messages。

    自己说过的话 -> assistant；对方说的话 -> user（带说话人前缀）。
    """
    names = cfg["names"]
    sys_prompt = SYSTEM_TEMPLATE.format(
        name=names[agent], other=names[other_of(agent)], topic=cfg["topic"]
    )
    msgs = [{"role": "system", "content": sys_prompt}]
    cond = cfg["condition"]

    if cond == "loop":
        history, bootstrap = transcript, BOOTSTRAP_A
    elif cond == "solo":
        history = [m for m in transcript if m["agent"] == agent]
        bootstrap = BOOTSTRAP_A
    elif cond == "shortmem":
        history = transcript[-cfg["shortmem_k"]:]
        bootstrap = None
    else:
        raise ValueError(cond)

    body = []
    for m in history:
        if m["agent"] == agent:
            body.append({"role": "assistant", "content": m["content"]})
        else:
            body.append({"role": "user", "content": f"{names[m['agent']]}：{m['content']}"})

    if not body:
        msgs.append({"role": "user", "content": bootstrap})
        return msgs

    if body[0]["role"] == "assistant":
        # 需要一条 user 起头，且不能打断 assistant/user 交替
        msgs.append({"role": "user", "content": bootstrap or NUDGE})
    msgs.extend(body)
    if body[-1]["role"] == "assistant":
        msgs.append({"role": "user", "content": NUDGE})
    return msgs


def other_of(agent: str) -> str:
    return "B" if agent == "A" else "A"


# ---------------------------------------------------------------- provider

class DeepSeekProvider:
    """直接打 REST 接口（不依赖 openai SDK：本机 openai 1.50.2 与 httpx 0.28 不兼容）。"""

    FATAL = {400, 401, 402, 403, 404, 422}   # 重试也没用

    def __init__(self, model: str):
        import requests

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            env_file = Path(__file__).with_name(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.strip().startswith("DEEPSEEK_API_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not key:
            sys.exit("缺少 DEEPSEEK_API_KEY（用 export 传入）")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}",
                                     "Content-Type": "application/json"})
        self.model = model

    def __call__(self, messages: list, cfg: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "thinking": {"type": "disabled"},   # 关思考：更便宜、更快、不污染转录稿
            "stream": False,
        }
        last_err = None
        for attempt in range(6):
            try:
                t0 = time.time()
                r = self.session.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300)
                if r.status_code != 200:
                    if r.status_code in self.FATAL:
                        raise SystemExit(f"API 拒绝请求 HTTP {r.status_code}: {r.text[:300]}")
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                j = r.json()
                u = j.get("usage", {})
                return {
                    "content": (j["choices"][0]["message"].get("content") or "").strip(),
                    "hit": u.get("prompt_cache_hit_tokens", 0) or 0,
                    "miss": u.get("prompt_cache_miss_tokens", 0) or 0,
                    "prompt": u.get("prompt_tokens", 0),
                    "out": u.get("completion_tokens", 0),
                    "latency_s": round(time.time() - t0, 2),
                    "finish_reason": j["choices"][0].get("finish_reason"),
                }
            except SystemExit:
                raise
            except Exception as e:  # 限流/网络抖动：指数退避
                last_err = e
                wait = min(60, 2 ** attempt) + random.random()
                print(f"    [retry {attempt+1}/6] {type(e).__name__}: {e} -> {wait:.1f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"6 次重试仍失败: {last_err}")


class MockProvider:
    """无 key 空跑：产出形状合理的假回复，用来验证日志/续跑/分析链路。"""

    FILLER = [
        "我想先把对方话里的那个词拆开看看，它似乎承担了太多意思。",
        "换个角度，如果把这个判断推到极端，结论会变得很奇怪。",
        "你举的那个例子我接受，但它依赖一个没有被说出来的前提。",
        "这里我倾向于退一步：也许我们问的问题本身就设错了边界。",
    ]

    def __call__(self, messages: list, cfg: dict) -> dict:
        time.sleep(0.01)
        n = len(messages)
        rnd = random.Random(n * 977 + messages[-1]["content"][:8].__hash__())
        text = f"（mock 第{n}条）" + "".join(rnd.sample(self.FILLER, 3)) + f"这是对第{n}条的回应。"
        pt = sum(len(m["content"]) for m in messages) // 2 + 20
        return {
            "content": text, "hit": int(pt * 0.9), "miss": int(pt * 0.1) + 10,
            "prompt": pt + 10, "out": len(text) // 2,
            "latency_s": 0.01, "finish_reason": "stop",
        }


# ---------------------------------------------------------------- 主循环

def load_done(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_markdown(records: list, cfg: dict, path: Path):
    names = cfg["names"]
    lines = [
        f"# 两个 LLM 的 {len(records)} 轮对话", "",
        f"- 条件：`{cfg['condition']}`　模型：`{cfg['model']}`　温度：{cfg['temperature']}",
        f"- 话题：{cfg['topic']}",
        f"- 轮数：{len(records)} 条消息（{len(records)//2} 轮来回）", "",
        "---", "",
    ]
    for r in records:
        lines += [f"### [{r['i']}] {names[r['agent']]}", "", r["content"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="loop", choices=["loop", "solo", "shortmem"])
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "mock"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rounds", type=int, default=100, help="轮数，1 轮 = 甲乙各说一次")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--shortmem-k", type=int, default=6)
    ap.add_argument("--topic", default="如果一个人可以随时重写自己的记忆，他还是同一个人吗？")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-usd", type=float, default=5.0, help="累计花费上限，超过即停")
    cfg = ap.parse_args()

    cfg.names = {"A": "甲", "B": "乙"}
    cfg_dict = vars(cfg).copy()
    total_turns = cfg.rounds * 2

    out_dir = Path(cfg.out) if cfg.out else Path("runs") / f"{cfg.condition}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "transcript.jsonl"

    records = load_done(jsonl)
    if records:
        print(f"续跑：已有 {len(records)} 条记录", flush=True)
    provider = MockProvider() if cfg.provider == "mock" else DeepSeekProvider(cfg.model)

    spent = sum(r.get("cost_usd", 0.0) for r in records)
    with jsonl.open("a", encoding="utf-8") as fh:
        for i in range(len(records), total_turns):
            agent = "A" if i % 2 == 0 else "B"
            messages = build_messages(agent, records, cfg_dict)
            res = provider(messages, cfg_dict)
            now = dt.datetime.now(dt.timezone.utc)
            peak = is_peak(now)
            cost = call_cost(cfg.model, res["hit"], res["miss"], res["out"], peak)
            spent += cost
            rec = {
                "i": i, "round": i // 2 + 1, "agent": agent, "content": res["content"],
                "prompt_tokens": res["prompt"], "cache_hit": res["hit"], "cache_miss": res["miss"],
                "completion_tokens": res["out"], "latency_s": res["latency_s"],
                "finish_reason": res["finish_reason"], "model": cfg.model,
                "condition": cfg.condition, "peak": peak, "cost_usd": round(cost, 6),
                "iso": now.isoformat(timespec="seconds"), "ts": now.timestamp(),
                "n_prompt_messages": len(messages),
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i+1:>3}/{total_turns}] {cfg.names[agent]} "
                  f"in={res['prompt']:>7} (hit {res['hit']:>7}) out={res['out']:>4} "
                  f"{res['latency_s']:>5.1f}s ${cost:.5f} 累计 ${spent:.3f}", flush=True)
            if spent > cfg.max_usd:
                print(f"达到花费上限 ${cfg.max_usd}，停止", flush=True)
                break

    write_markdown(records, cfg_dict, out_dir / "transcript.md")
    meta = {
        "config": {k: v for k, v in cfg_dict.items() if k != "names"},
        "names": cfg.names,
        "n_messages": len(records),
        "totals": {
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "cache_hit": sum(r["cache_hit"] for r in records),
            "cache_miss": sum(r["cache_miss"] for r in records),
            "completion_tokens": sum(r["completion_tokens"] for r in records),
            "usd": round(spent, 4),
            "wall_clock_s": round(sum(r["latency_s"] for r in records), 1),
        },
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {len(records)} 条 -> {out_dir}\n累计 ${spent:.4f}  "
          f"in {meta['totals']['prompt_tokens']:,} tok "
          f"(cache hit {meta['totals']['cache_hit']:,}) / out {meta['totals']['completion_tokens']:,} tok",
          flush=True)


if __name__ == "__main__":
    main()
