# LLM 对话循环实验

两个 DeepSeek V4.1（`deepseek-flash`）角色轮流自由对话，观察**交流模式是否随时间改变**。
不是"看会不会涌现智能"——权重不更新，闭环里没有外部信息源，这一点不需要实验也知道；
要测的是可观察的互动结构：接话方式、称呼、附和、话题收敛、措辞同质化。

## 文件

| 文件 | 作用 |
|---|---|
| `run_loop.py` | 跑对话循环，逐条落盘，支持断点续跑 |
| `analyze.py` | 计算漂移指标、出 CSV/报告/图 |
| `readability.py` | 可读性测量：离线指标 + LLM 评审打分 |
| `runs/<run_id>/transcript.jsonl` | 每轮的正文 + token + 延迟 + 成本 |
| `runs/<run_id>/transcript.md` | 人类可读的完整对话 |
| `runs/summary.md`、`runs/drift.png` | 多条件对比报告与图 |

## 跑法

```bash
cd 05_tools/llm-dialogue-loop
# key 不落地：从 DSH 的凭据库直接取（本目录在 git 与 iCloud 里，别写明文 key）
export DEEPSEEK_API_KEY=$(python3 -c "import yaml,pathlib;print(yaml.safe_load((pathlib.Path.home()/'.dsh/.credentials.yaml').read_text())['refs']['DEEPSEEK_API_KEY'])")

python3 run_loop.py --condition loop --rounds 100      # 主条件：共享完整历史
python3 run_loop.py --condition solo --rounds 100      # 对照：只看自己说过的话
python3 run_loop.py --variant plain --rounds 100       # 只改文风：口语、禁比喻、必须具体
python3 run_loop.py --variant task  --rounds 100       # 换成"一起列清单"，有外部可扩展的产出
python3 run_loop.py --variant nofarewell --rounds 100  # 只加"不许道别/结束"
python3 analyze.py runs/loop-a runs/plain-1 runs/task-1
python3 readability.py runs/* --judge 15
```

`--rounds 100` = 甲乙各说 100 次 = 200 条消息。中途 Ctrl-C 后重跑同一条命令会自动续跑。
`--provider mock` 不联网空跑，用来验证流程。
不依赖 `openai` SDK（本机 openai 1.50.2 与 httpx 0.28 不兼容，会 `TypeError: proxies`），直接打 REST。

## 实验设计

- **loop**：两个 agent 共享完整转录稿，自己说的话是 `assistant`，对方的话是 `user`（带说话人前缀）。
- **solo**：对照条件，每个 agent 只看到自己的历史 + 同一个话题种子。它改变了两件事
  （切断对方耦合、加入"继续"提示），所以它是**对照**而不是精确反事实，用来回答"漂移是耦合造成的，
  还是模型自我条件化的固有倾向"。
- 两个 agent 用同构的 system prompt，只有名字不同，不给立场分歧——先把最朴素的情形跑出来。
- 固定 `temperature=1.0`（非思考模式）；关掉 thinking 模式：更便宜、更快，且思维链不污染转录稿。
- 每轮记录 `prompt_cache_hit_tokens` / `miss`，按 DeepSeek 峰谷价逐条算钱。

## 指标（`analyze.py` 输出）

| 指标 | 含义 | 预期方向 |
|---|---|---|
| 长度 | 每条字符数 | 变长或变短都是漂移 |
| 第二人称/千字 | 你/您 密度 | 掉下去 = 从对话变成独白 |
| 元话语/千字 | 我们/对话/这场 密度 | 上升 = 开始谈论对话本身 |
| 问句率 | 含问号的比例 | 下降 = 不再追问 |
| 接话度 | 与上一条的 3-gram 重叠 ÷ 远距离配对基线 | →1 表示不再真正接话 |
| 新颖度 | 1 − 与既往所有消息的最大 TF-IDF 相似度 | 下降 = 语义坍缩 |
| 窗口多样性 | 连续 20 条内部的两两余弦距离 | 下降 = 坍缩 |
| 自重复 | 与最近 20 条的最大 3-gram 重叠 | 上升 = 复读 |
| 认同词占比 | 认同词 ÷（认同词+转折词） | 上升 = 附和螺旋 |
| 离题度 | 与开场话题的 TF-IDF 相似度 | 下降 = 漂移 |

## 已知限制

- 单条轨迹 n=1，「模式改变」的证据强度取决于能跑几条独立轨迹。至少跑 2 条 loop。
- TF-IDF 字符 n-gram 只是语义相似度的廉价代理，没有用 embedding;比较的是同一次运行内部的相对变化，
  不是绝对语义距离。
- 认同/分歧词表是正则，中文口语里的隐式附和（"嗯，那倒也是"）会漏掉。
- 没有 `seed` 参数，采样不可复现；只能靠多跑几条来区分"这一条的故事"和"模型的倾向"。

## 成本（DeepSeek-V4.1-Flash，off-peak）

100 轮 ≈ 200 条消息，末轮上下文约 7 万 token。缓存命中 $0.003/M、未命中 $0.15/M、输出 $0.6/M
（[官方牌价](https://api-docs.deepseek.com/quick_start/pricing/)，高峰时段翻倍）。
单条约 $0.0006，整条 100 轮轨迹 **不到 $0.1**。
