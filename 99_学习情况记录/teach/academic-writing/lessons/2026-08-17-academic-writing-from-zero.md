---
aliases:
  - LN905 Academic Writing course replay
tags:
  - teach
  - learning/academic-writing
date: "2026-08-17"
source_type: "teach"
topic: "LN905 Academic Writing"
status: "active"
course: "LN905"
record_type: "foundation-course"
phase: "teaching-practice"
current_block: "1A"
---

# Academic Writing 从零重学

> [!summary] 教学约定
> 把学习者当作没有上过 Academic Writing。每次只教一个完整小点，立即用一道短构造题检查；讲评后再进入下一点。从第二个教学块开始，测验会混入一道到期旧题，确保不仅当场听懂，也能过一段时间再次调用。

<!-- student-os:mentor-brief:start
今日 principal：建立 Academic Writing 的最小心智模型：问题 → 自己的答案/claim → evidence → reasoning，并用新情境完成一次三句应用。
重点技能 IDs：AW-16；随后按先修顺序进入 AW-01、AW-02、AW-03–AW-17。
starting evidence：用户明确要求按零基础重新授课；现有 AI 整理稿、detailed plan 和历史成文不作为可以跳课的掌握证据。
complete output：学完已激活的 Academic Writing 内容，在新题上独立完成 question analysis、thesis、structure、source decisions、argument paragraphs、opening/closing 与 criteria-led revision，并通过累计迁移应用。
teaching chain：最小写作模型 → question analysis → thesis → top-down structure/topic sentences → source reliability/use → paraphrase/summary/quotation/fidelity → synthesis → paragraph argument/Toulmin → writer voice/claim strength/cohesion → introduction/conclusion → revision → cumulative application。
allowed support：自然中文解释、课件例子、worked contrast、简短句框和逐步淡出的支架；测验先允许中文证明理解，再逐渐转入英文产出。
review schedule：一个概念在即时检查中首次成立后，进入 +1、+3、+7 教学块的 review queue；每轮最多加入一道到期旧题，结课做累计应用。旧题换 topic 或材料，不重复原句。
silent callbacks：本基础课程不使用完整输出前隐藏的 callback；会提前说明有旧知识抽查，但不提前泄露具体题目。
hint ladder：重显必要材料 → 问缺失的是答案、证据还是关系 → 给一个不同 topic 的正反对比 → 简短 model 后换材料重测。
feedback priority：先判断概念和逻辑关系是否成立，再看 source meaning，最后才处理语言；当前阶段不因英文或术语阻挡理解。
completion evidence：每个重点先通过当前构造题，再在到期旧题和最终累计应用中无提示复现；仅听过或看过不算掌握。
writeback rule：每次教学回复前追加 learning log，并更新本文件 review queue；只有新的可观察证据才写入 LN905 Skill Bank。课程测验不创建 checkbox。
stop boundary：完成已激活课程内容和累计应用后停止并总结；在此之前不切换到 credit-scoring essay 代写或 Paper A/B 训练。
student-os:mentor-brief:end -->

## 课程地图

1. 学术写作的最小模型：问题、答案、证据、reasoning。
2. 读懂题目：task、parts、restrictions、definitions、歧义、观点与结构。
3. 写 thesis：直接回答、specific、focused，并让路线可见。
4. 自上而下搭结构：四种 structure、paragraph function 与 topic sentence。
5. 选择并忠实使用来源：reliability、intended use、paraphrase、summary、quotation 与 citation。
6. Synthesis：让多份来源共同支持、限定或挑战自己的判断。
7. 正文小论证：Description → Analysis → Evaluation，以及 Toulmin 六部分。
8. Academic voice、cohesion 与 evidence-strength calibration。
9. Introduction、conclusion 与 criteria-led revision。
10. 新题累计应用，再迁移到真实 assignment。

> [!note] 当前材料边界
> Week 4 只纳入已实际授课的 Lesson 1 Making Claims。Passive voice、nominalisation 和 concise writing 暂不作为已学课程内容。

## 复习机制

- 新概念：讲解和例子后立即做一道构造题。
- 首次证明理解后：在第 `+1、+3、+7` 个教学块重新出现。
- 从 1B 开始：一次测验通常包含一个当前应用和最多一个到期旧题。
- 旧题答错：用一个新对比讲清，放回 `+1`；不反复重写同一句。
- 这些只是同一课程中的学习证据，不产生任务或完成状态。

## Review queue

| AW ID | 内容 | 最近证据 | 下次复习轮次 | 阶段 |
|---|---|---|---|---|
| AW-16 | `question → claim → evidence → reasoning` 最小链 | 等待本轮即时检查 | 回答后决定 | 正在教学 |

## 给续学 Codex 老师的 init prompt

你负责继续这门 Academic Writing 零基础课程，而不是推进当前 credit-scoring essay。开始前读取根目录 `AGENTS.md`、`CLAUDE.md`、本文件、[[99_学习情况记录/teach/academic-writing/MISSION|MISSION]]、[[99_学习情况记录/teach/academic-writing/RESOURCES|RESOURCES]]、[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing|Academic Writing 综合讲稿]] 与 [[99_学习情况记录/teach/LN905 Skill Bank|Skill Bank]]，从 learning log 和 review queue 的最新状态继续。

每次只教一个 5–10 分钟的小块。先用中文说明它解决什么写作问题，再给一个课件例子或短对比，然后出一道不能靠措辞猜中的构造题。从第二教学块开始检查 review queue，并在本轮题里最多混入一道到期旧题。收到回答后先反馈，再决定继续、换材料重测或简短重教。每次回复前先追加 learning log；没有新的学习者证据时不要更新 Skill Bank。

## Block 1A｜学术写作到底在做什么？

学术写作不是堆事实，也不是把“作者 A 说……作者 B 说……”排在一起。它最小的推理链是：

`问题 → 你的答案（claim）→ evidence → reasoning`

- **Claim**：你希望读者接受的答案或主张。
- **Evidence**：数据、研究或其他可靠资料。
- **Reasoning**：你解释这份证据为什么让 claim 更可信。

只有 claim、没有 evidence，是没有依据的意见。只有 evidence、没有自己的 claim，是资料汇总。有 claim 和 evidence、却没有 reasoning，读者仍要替你完成最关键的推理。

> [!example] 课件中的 Social Media 题目
> 题目问：很多人使用 social media，这本身是否值得担忧？
>
> “很多学者研究了 social media”只是在介绍话题，没有给答案。
>
> “用户规模本身不足以证明普遍伤害；如果研究发现平均影响很小，而风险集中在特定使用方式和群体，担忧就应该集中在那些条件上”才出现了答案、证据和它们之间的关系。

## 即时检查 1A

> [!question] 教学情境（假设数据）
> **题目：**提高女性员工比例，是否足以消除职场性别不平等？
>
> 一项对 60 家公司的假设研究发现：
>
> - 女性占比超过 40% 的公司，晋升差距并不比其他公司小；
> - 公开晋升标准的部门，晋升差距更小。

请只写 **三句中文**：

1. 第一句直接回答题目。
2. 第二句准确使用研究 evidence。
3. 第三句解释这份 evidence 为什么支持你的答案。

暂时不用写英文，也不用定义术语。

<!-- student-os:learning-log:start -->

### 2026-08-17 16:54 BST

- **学习者输入**：明确纠正上一版：把自己当作没上过 Academic Writing 的小白；根据仓库笔记重新上完整课程；先教一部分再考一部分，并定期抽查很早以前的内容。
- **AI teaching/feedback**：承认旧目标错误，停用 essay-first 记录。建立 foundation course、review queue 与 `+1/+3/+7` 复习节奏。本轮教学 Academic Writing 最小模型，并给出一题三句构造检查；没有假设任何既有掌握。
- **当前 skill-map 位置**：Block 1A；`AW-16` 的最小 top-down reasoning chain。尚无本课程新证据。
- **单一下一动作**：学习者根据 Gender 假设研究写三句中文，分别承担 claim、evidence 与 reasoning。
- **internal continuation decision**：若三部分清楚且 evidence 未被改写，记录首次教学证据并把 AW-16 放入 +1/+3/+7 queue；下一块教学 topic 与 task 的区别，并混入一道 1A 旧题。若缺一部分，用 Climate Change 对比讲清后换材料重测，不要求重复原句。

<!-- student-os:learning-log:end -->
