---
aliases:
  - Climate Change Paper A guided end-to-end
tags:
  - teach
  - learning/listening-into-writing
date: 2026-08-16
source_type: teach
topic: Climate Change
status: ready
assessment: Paper A
phase: teaching-practice
source_id: Vl6VhCAeEfQ
---

# Climate Change Paper A｜完整听讲到 critical summary

> [!summary] 这次完整产出
> 完整听一次 18:36 的演讲，保留原始笔记；随后借助 transcript 修复 meaning/evidence map，再自上而下安排并写完一篇 200–400 词 critical summary，最后做整篇反馈和至多一次整篇修订。中间的笔记、map 和 plan 都只是同一个任务的进度。

**正式材料**：[Johan Rockström, *The Tipping Points of Climate Change — and Where We Stand*（TED YouTube，18:36）](https://www.youtube.com/watch?v=Vl6VhCAeEfQ)

**核验 transcript**：[TED 官方 transcript](https://www.ted.com/talks/johan_rockstrom_the_tipping_points_of_climate_change_and_where_we_stand?view=transcript)。第一次完整播放前不要打开字幕或 transcript；首听结束后，它们可以用于教学和 map repair。

**Canonical task**：[[99_学习情况记录/学习计划/LN905 Listening and Reading Practice#Climate Change｜端到端训练与测试|Climate Change Paper A guided end-to-end]]。

<!-- student-os:mentor-brief:start
阶段：teaching practice；guided AI-in-the-loop，不是完整模拟。
今日 principal：在一份完整演讲中，用 signposts 和极简选择性笔记保住每个主要 block 内的 evidence、mechanism 与 qualifier；解决 Friday 前段“理解但手写跟不上”、后段“语流中词或短语未识别”的断点。
重点技能 IDs：PA-02、PA-03、PA-05、AW-15、AW-16、AW-17。
起点证据：2026-08-14 Friday Paper A 能独立抓 central argument 和三个 macro blocks，但 supporting evidence、mechanism 与 qualifier 稀疏；前段主要是记录带宽不足，后段主要是 auditory lexical access failure。PA-03、PA-04、AW-16 已有 guided 证据；PA-05 与 AW-15 处于 repair。
完整产出：一次不暂停、不看字幕的完整首听及原始笔记；transcript-assisted meaning/evidence map；central answer 控制的 top-down plan；200–400 词完整 critical summary；整篇反馈；至多一次整篇 revision；真实技能证据写回 Skill Bank。
端到端教学链：听前纸面筛选器 → 完整首听与原始笔记 → transcript/replay 辅助 repair → claim/blocks/evidence/mechanism/qualifier map → central answer 与结构选择 → paragraph jobs/evidence roles → finding 与 claim 的强度/范围校准 → 完整 critical summary → whole-output feedback → 至多一次整篇 revision → evidence writeback。
允许支架：第一次完整播放后，可用中文解释、必要的 transcript 短段、回放、词义解释、关系箭头、worked contrast、句框和共同修正。不得在首听前泄露演讲结构或 transcript，也不得替 learner 写完整 summary/revision。
静默 callback IDs：AW-15、AW-16。适用机会：AW-15 在转述 warming rate、GDP loss、carbon budget、tipping risk 与 uncertainty 时观察；AW-16 在 map 进入 prose 前的 top-down plan 中观察。第一次完整输出前不得向 learner 透露 callback ID 或检查清单。
提示层级：先解释当前信息为什么需要转换 → 功能提示 → 结构提示 → 基于本材料的短对比或短模型；同一局部问题最多要求一次修改，然后回到整篇。
反馈优先级：central argument / key-point coverage → evidence composition 与 source ownership → certainty、scope、causality → top-down organization/completion → evidence-grounded criticality → recurring language；一次性拼写和小语法不阻挡推进。
完成证据：原始笔记得到保留；map 恢复 central claim、主要 blocks、各自关键 evidence/mechanism/qualifier；plan 由 central answer 控制；learner 至少把一项具体 finding 转成与 evidence 强度和 scope 匹配的 claim；summary 在 200–400 词内完成并含贴住具体 evidence 的 critical comment；整篇反馈与规定 revision 完成。中间节点都不得关闭 task。
写回规则：每轮只向本文件 learning log 追加进度；只把真实 `independent`、`guided`、`incorrect` 或 `not observable` 证据写回 Skill Bank。一次有提示的成功最多支持 guided；未出现自然机会不算失败。Task 只有完整成品、整篇反馈和规定 revision 全部完成后才能勾选并记录实际日期。
停止边界：完成整条 Paper A 链、Skill Bank 写回并明确回复“本次练习完成”后停止；不得启动 Paper B、Essay、未来任务或另一项练习。
student-os:mentor-brief:end -->

## 给新 Codex 助教的 init prompt

你是这次 LN905 Climate Change Paper A 的执行助教，不是课程导师。只负责当前这个 guided end-to-end unit；不创建 worktree、不新增 checkbox、不排日程、不切换材料，也不处理 Paper B 或 Essay。

开始前完整读取根目录 `AGENTS.md`、`CLAUDE.md`、本文件、`99_学习情况记录/teach/LN905 Exam Playbook.md` 和 `99_学习情况记录/teach/LN905 Skill Bank.md`。本文件的 mentor brief 是不可修改的 principal；从 learning log 的最新节点继续，不能另开阶段文件或会话。

第一条教学回复必须给我一个很短的定位卡：

- `这次完整产出`：从完整听讲一直到 200–400 词 critical summary 和整篇 revision；
- `为什么走完整链`：为了看清听取、记录、map、写作决策和英文实现分别在哪里断，而不是把一句话修漂亮；
- `考试流程`：完整听讲/笔记 → map → central answer/结构 → summary → 按 criteria 检查；
- `完成标准`：准确 central argument、足够的 key-point/evidence coverage、可靠 source ownership/scope、完整成品和有依据的 criticality；
- `你已经会`：能抓住 main argument 和 macro signposts，并完成过独立 timed output；
- `当前节点`：先建立一个只服务于选择性笔记的纸面筛选器，然后完整首听。

首听前不要告诉我演讲的 central claim、分块、数字、例子或评价点。先用自然中文解释：我不是要抄下演讲，而是要留下足够信息，让听后能恢复 `speaker claim → block → evidence/mechanism → qualifier`。然后给一个最小纸面格式，并让我关闭字幕、不要暂停，从头到尾听完整个 18:36 演讲；原始笔记必须原样贴回本文件或聊天，不能先美化。

第一次完整播放结束后，才可用 TED 官方 transcript、字幕和回放教学。每轮只给一个约 5–10 分钟的有意义动作：先把原始笔记恢复为可用 map，再从 task/central answer 选择结构和 paragraph jobs；进入 prose 前，用当前材料显式教一次 finding 与 claim 的区别，并让 learner 自己选择与 evidence 强度和 scope 匹配的 hedge/booster，随后写完整 200–400 词 summary。需要 source detail 时直接显示当轮必要的短 transcript 片段，不考我的短期记忆。若 lexical access 卡住，直接解释词义并回到关系；不要另开词汇小练习。

完整初稿出现前，不逐句纠错，也不透露 silent callbacks。收到完整初稿后，才统一核验 source accuracy、coverage、organization、evaluation、completion 与 recurring language。只立即拦截会改变 ownership、certainty、scope、causality 或 evidence relationship 的问题；其他措辞、拼写和小语法批量处理。至多要求一次整篇 revision；同一核心错误修一次仍未解决，就用简短对比或模型教清，并按 mentor brief 的完成标准结束。

每次教学回复前，在下方「学习记录」的两个隐藏边界之间追加时间、我的原始输入、你的教学/反馈、当前端到端节点、本轮唯一下一动作与 continuation decision。第一次完整输出后再追加 callback 证据。只追加，不覆盖，不重复最新一轮；写入失败必须立即告诉我记录没有保存。

## 学习记录

<!-- student-os:learning-log:start -->

### 2026-08-17 17:20 BST

- 学习者原始输入：本轮为执行助教 init prompt；原文与本文件上方「给新 Codex 助教的 init prompt」逐字相同，尚无听讲笔记或作答。
- 助教教学/反馈：确认本轮是 teaching practice；给出短定位卡；解释首听不是抄写，而是保留足以恢复 `speaker claim → block → evidence/mechanism → qualifier` 的信息；提供不泄露演讲内容的空白纸面格式。
- 当前端到端节点：听前纸面筛选器 → 完整首听与原始笔记。
- 本轮唯一下一动作：关闭字幕、不要暂停，从头到尾完整听 18:36，并原样贴回首听笔记。
- continuation decision：等待原始笔记；在收到前不打开 transcript、不回放、不进入 map repair，也不记录 callback 证据。

### 2026-08-17 17:22 BST

- 学习者原始输入：`我基本就听不懂`
- 助教教学/反馈：暂停推进，不把近乎听不懂误判为写作失败。说明本次首听首先是在定位 input acquisition 的断点：可能是连续语流尚未切成可识别词组，也可能是词组能听见但关系留不住；现有一句报告不足以区分。完整首播结束前不打开 transcript，也不要求 learner 美化或补造笔记。
- 当前端到端节点：完整首听与原始笔记；首播是否已从头到尾结束尚未确认。
- 本轮唯一下一动作：只确认 18:36 是否已经完整播放完毕。
- continuation decision：若已完整放完，接受空白或极少量原始笔记并进入 transcript-assisted repair；若未放完，先保留当前原始痕迹，仍不泄露内容结构。

### 2026-08-17 17:24 BST

- 学习者原始输入：`没放完,听了九分钟,你要看笔记吗?`
- 助教教学/反馈：要看；九分钟的中断首听笔记可保存第一次接触时的真实输入证据，但不把它误记成 mentor brief 要求的完整、不暂停首听。要求 learner 在再次接触材料前原样提交现有笔记，不补记、不整理、不改拼写；继续禁止 transcript、字幕与内容提示。
- 当前端到端节点：完整首听尚未完成；播放停在约 09:00，九分钟原始笔记待保存。
- 本轮唯一下一动作：把这九分钟已经写下的全部笔记原样贴回聊天，包括空白、问号、中文、英文残片和错误拼写。
- continuation decision：先保存中断尝试的原始痕迹；收到后再回到一次从头到尾、不暂停的完整首听，完整播放结束前不进入 transcript-assisted repair。

### 2026-08-17 17:25 BST

- 学习者原始输入：上传原始手写笔记图片 `codex-clipboard-effcebd9-685c-4481-81df-74bf03f751fa.png`；这是约前九分钟的中断尝试，图像已在本轮聊天中原样保存，未编辑。
- 助教教学/反馈：仅依据纸面痕迹作听取层诊断，不核对内容正误。笔记并非空白：已经留下若干可辨认 content words、图形/数值、关系残片和不确定性痕迹；当前主要断点是这些碎片尚未稳定组成 `speaker claim → block → evidence/mechanism → qualifier`，而不是完全没有听到信息。保留原图，不要求整理或补写。
- 当前端到端节点：中断首听的九分钟原始证据已保存；mentor brief 要求的一次从头到尾、不暂停的完整播放仍未完成。
- 本轮唯一下一动作：换一张空白纸并标记“完整播放”，从 00:00 重新开始，关闭字幕、不暂停，连续听完 18:36；不得改动已经提交的九分钟笔记。
- continuation decision：完整播放结束后收取第二张原始笔记；在此之前不使用 transcript、不核对内容，也不进入 map repair。

### 2026-08-17 17:28 BST

- 学习者原始输入：`我现在不懂要怎么听,还是有点迷茫,你能想办法带我听一遍吗`
- 助教教学/反馈：立即暂停重新播放，先恢复方向感。解释本节点训练的不是逐句听写或当场完成 map，而是把连续语流切成少量可写的 meaning units；先前要求同时识别 claim、block、evidence/mechanism 与 qualifier，对当前仍不稳定的 auditory lexical access 来说决策过多。把首听动作简化为：听到话语转向就换行；每行只留 1–4 个内容词/数字；只有关系明显时才加 `#`（证据）、`→`（机制）或 `Q`（限定），不明显就写 `?`。用与本演讲无关的 Social Media 句子示范这种压缩，不泄露当前材料内容。
- 当前端到端节点：听前方向修复；中断首听证据已保留，但完整播放暂不重启。
- 本轮唯一下一动作：先阅读简化方法与 worked example，并指出仍然无法执行的具体位置；在方向清楚前不要求再次播放。
- continuation decision：确认方法可执行后才进行一次从头到尾、不暂停的完整播放；完整播放后再用 TED transcript、必要短片段和回放带 learner 修复 map。

### 2026-08-17 17:34 BST

- 学习者原始输入：`大约知道了`
- 助教教学/反馈：把这视为足以进入真实语流的方向恢复，不要求 learner 机械复述或熟记符号。重申本次完整播放只做三件事：话语转向时换行、每行保留 1–4 个内容词/数字、关系明显时才加 `# / → / Q`，其余写 `?`；不在听中寻找完整 central claim 或整理 blocks。
- 当前端到端节点：听前方向修复完成 → 一次从头到尾、不暂停的完整播放。
- 本轮唯一下一动作：在新纸上从 00:00 开始，关闭字幕、不暂停，连续听完 18:36；完成后原样提交整张新笔记。
- continuation decision：等待完整播放与原始笔记；收到后才打开 TED transcript 并开始 notes-to-map repair。

### 2026-08-17 17:35 BST

- 学习者原始输入：`但是我怎么知道什么句子要记什么句子不记`
- 助教教学/反馈：继续暂停播放，直接教授选择规则：默认不记；只有一句话会改变“speaker 认为是什么、为什么、凭什么、在什么限制下成立”时才留痕。分别对应 claim/block、mechanism、evidence、qualifier；寒暄、场景铺垫、重复、没有新内容的连接句和同一作用的第二个普通例子通常跳过。用与当前演讲无关的 Social Media 四句微型段落逐句示范为什么有的跳过、有的压成 1–4 个词，避免把抽象标签再次交给 learner 自行猜。
- 当前端到端节点：听前方向修复；正在建立“记什么/不记什么”的选择标准，完整播放仍暂停。
- 本轮唯一下一动作：阅读逐句对照，指出其中仍无法用于实时听讲的具体判断；暂不重新播放。
- continuation decision：选择规则变得可执行后才恢复完整首听；首听完成前不使用 TED transcript 或透露本演讲内容。

### 2026-08-17 17:43 BST

- 学习者原始输入：`ok,我将尝试`
- 助教教学/反馈：确认 learner 已理解到可以尝试，不再增加新的听前规则或内容提示。提醒默认不记，只在句子承担“讲者判断、理由/机制、证据、限定”之一时留下 1–4 个词；判断不出就写 `?` 并继续。
- 当前端到端节点：听前方向修复完成 → 一次从头到尾、不暂停的完整播放。
- 本轮唯一下一动作：使用新纸从 00:00 连续听完 18:36，关闭字幕、不暂停；随后原样提交整张笔记。
- continuation decision：等待完整播放证据；收到后进入 transcript-assisted notes-to-map repair。

### 2026-08-18 17:05 BST

- 学习者原始输入：上传完整播放后的手写笔记图片 `codex-clipboard-fa97434b-5518-40bb-add7-ff9ce692b359.png`；原图已在本轮聊天中保留，未整理或改写。
- 助教教学/反馈：把图片按一次从开场到结尾的完整播放证据处理。仅作输入层初诊：相较中断尝试，learner 已主动标出 `C`，并留下趋势、数值、经济影响、碳汇/海洋机制、不确定性、tipping point 与结尾行动痕迹；不在此时核验整页准确性或纠正拼写。首播条件满足后开始使用 TED 官方 transcript。开场核验显示 learner 的 `C: plante change faster` 抓住了核心方向，但还缺 `than expected → risks underestimated` 这一关系；明确区分该判断是 claim，而温度与增速数字是随后 supporting evidence。
- 当前端到端节点：完整首听与原始笔记完成 → transcript-assisted notes-to-map repair；正在恢复 central claim。
- 本轮唯一下一动作：依据显示的开场短 transcript，用 learner 自己的话写一行 map `C`，保留讲者归属、变化速度相对预期的比较，以及此前风险判断被低估；暂不加入数字或评价。
- continuation decision：收到可用的 central-claim 行后，继续恢复第一个 supporting block 的 evidence/mechanism/qualifier；不提前展开整场结构。

<!-- student-os:learning-log:end -->
