---
aliases:
  - LN905 credit-scoring essay guided writing
tags:
  - teach
  - learning/academic-writing
date: "2026-08-17"
source_type: "teach"
topic: "LN905 Academic Writing"
status: "active"
course: "LN905"
record_type: "guided-end-to-end"
phase: "teaching-practice"
---

# Credit-scoring essay：从题目到完整论文

> [!summary] 本次方向
> 用当前正式 essay 走完一条真实学术写作链。已有 question、thesis、plan 和 sources 是起点材料；每个关键判断仍由学习者亲自完成，不能由 AI 静默代做。

**Canonical task**：[[07_Programme/01_LN905_LSE-language-class/02_LN905_essay/LN905 Essay#执行清单|LN905 Essay first-draft 与 revision 任务链]]。

<!-- student-os:mentor-brief:start
今日 principal：把当前 credit-scoring question 恢复为学习者可控制的自上而下写作过程，并最终完成约 2,000 词 first draft 与一次整篇 revision。
重点技能 IDs：AW-16 自上而下程序；AW-01 审题；AW-02 thesis；AW-15 source meaning fidelity；AW-17 evidence–claim 强度校准。
starting evidence：已有正式 question、working thesis、thematic structure、五个正文 topic sentences 和九篇来源；AW-16 = guided，AW-15 = repair，AW-01/AW-02/AW-17 = new。
complete output：学习者亲自完成约 2,000 词 essay，使用至少六篇 academic sources，包含直接 thesis、writer-led body paragraphs、conclusion 与 bibliography；随后按 criteria 完成一次整篇 revision。
end-to-end teaching chain：question analysis → provisional thesis → structure/topic sentences → evidence roles/selection → body-paragraph argument → introduction/conclusion → criteria-led whole-output feedback/revision。
allowed support：中文解释、课件例子、短对比、关系箭头、局部 sentence frame、协作讨论与来源回看；AI 不得替学习者选择 task、thesis、paragraph jobs 或 evidence roles。
silent callback IDs：AW-12、AW-13；首次完整输出前不得向学习者透露 callback ID 或检查清单。
hint ladder：先重显题目/材料 → 指出应判断的关系 → 给一个不复用当前答案的正反对比 → 给局部框架 → 一次失败后用简短 model 教清并回到整合应用。
feedback priority：task/meaning/source accuracy → reasoning/structure/completion → recurring language → isolated wording or grammar。
completion evidence：学习者完成所有关键决策、完整成稿和一次整篇 revision；来源的 ownership、certainty、scope 与 causality 经回查未被改变。
writeback rule：每轮先追加 learning log；只有可观察证据才更新 LN905 Skill Bank；canonical task 只按真实完成状态关闭。
stop boundary：完整成稿与规定的一次整篇 revision 后停止；不另开 remedial drill、不换材料、不安排未来任务。
student-os:mentor-brief:end -->

## Orientation card

- **这次完整产出**：当前 LN905 credit-scoring essay 的约 2,000 词完整稿，以及一次按 criteria 排序的整篇修改。
- **为什么走完整链**：学术写作的每一步都限制下一步。题目决定 thesis，thesis 决定段落工作，段落工作决定证据，证据强度决定 claim；只练漂亮句子不会形成可迁移的写作能力。
- **考试流程**：审题 → thesis → structure/topic sentences → sources → paragraph argument → introduction/conclusion → criteria check。
- **完成标准**：回答题目；至少六篇 academic sources；来源含义准确；段落由自己的 claim 推进；analysis/evaluation 可见；全文逻辑与语言满足 Assignment Marking Criteria。
- **你已经会**：已有整体答案、thematic route、五个正文功能和九篇来源，也曾在支持下完成过“总答案 → 段落功能 → evidence role → prose”。这说明不需要从“什么是 essay”重新学。
- **当前起点**：Stage 1 — Question analysis。先证明自己看见题目要求的判断，再核对现成 thesis；已有 plan 不能代替这项能力证据。

## 完整路线与当前位置

`→ Question analysis` → `Thesis` → `Structure` → `Sources` → `Body argument` → `Introduction/Conclusion` → `Revision`

## 给续学 Codex 助教的 init prompt

你是当前 LN905 Academic Writing teaching-practice 的执行助教。只在本 task 中续做这篇 credit-scoring essay；不创建新 checkbox、不排日程、不换题，也不替学习者写出关键判断或完整段落。

开始前完整读取根目录 `AGENTS.md`、本文件、[[99_学习情况记录/teach/LN905 Exam Playbook|LN905 Exam Playbook]]、[[99_学习情况记录/teach/LN905 Skill Bank|LN905 Skill Bank]]、[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing|Academic Writing 综合讲稿]] 和当前 [[07_Programme/01_LN905_LSE-language-class/02_LN905_essay/LN905 Detailed Essay Plan|Detailed Essay Plan]]。从 learning log 最新节点继续。

每次回复前，先在 learning-log 边界内追加学习者原始输入、AI teaching/feedback、当前 skill-map 位置、唯一下一动作和 continuation decision；只追加，不覆盖，不重复。每轮用自然中文先解释当前真实写作问题、为什么这一步能解决它、它怎样改变现有信息，以及它会解锁下一步什么决定，然后只给一个约 5–10 分钟的动作。

首次完整 draft 前不逐句纠错，也不透露 silent callbacks。完整稿出现后才按 mentor brief 的优先级做整篇反馈，并至多要求一次整篇 revision。只有可观察证据才写回 Skill Bank；完整输出与 revision 都完成后才关闭 canonical task。

## Stage 1｜Question analysis

课件先问的不是“这个主题讲什么”，而是 **What do you have to do?**。你的题目不是要求罗列 complex machine learning 与 logistic regression 的优缺点，而是要求作出一个有证据标准的判断：预测收益在多大程度上足以接受可解释性的下降。

> [!quote] 当前正式题目
> *To what extent do the predictive gains of complex machine-learning models over logistic regression justify reduced interpretability in consumer credit scoring?*

`To what extent` 允许答案是“只在某些条件下”，但不能停在 `it depends`；必须说明依赖什么。`justify` 也不是“存在一些好处”，而是要求你提出判断标准，比较收益是否足以抵偿代价。

### 当前唯一动作｜约 5–10 分钟

先不要看现成 working thesis。请用 **2–3 句中文**解释这道题真正要求你作出什么判断。答案中自然说明：比较对象是什么、什么 evidence 才能判断 `justify`，以及 `to what extent` 为什么不能写成一份普通优缺点清单。

<!-- student-os:learning-log:start -->

### 2026-08-17 16:40 BST

- **学习者输入**：Academic Writing 内容已经更新，希望根据讲义和 AI 生成的综合讲稿，一步一步学习学术写作。
- **AI teaching/feedback**：核对 Weeks 1–4 综合笔记、Week 1 原始讲义、当前 essay project、detailed plan 与 Skill Bank。把本次定位为 teaching practice；说明完整七步链、已有能力和当前起点。本轮只教学 question analysis，尚不判断学习者是否掌握。
- **当前 skill-map 位置**：整条链为 `AW-16`；当前节点为 `AW-01`。已有计划产物，但没有独立 question-analysis 证据。`AW-15` 的来源含义准确性风险将在后续 source/paragraph 阶段贯穿处理。
- **单一下一动作**：学习者用 2–3 句中文解释当前题目要求的判断，包含比较对象、`justify` 的证据标准与 `to what extent` 对答案形式的限制。
- **internal continuation decision**：收到回答后先判断 task、scope 与 required judgment 是否准确；若成立，进入 provisional thesis；若不成立，用一次简短对比教清后回到同一完整链，不另开练习。

<!-- student-os:learning-log:end -->
