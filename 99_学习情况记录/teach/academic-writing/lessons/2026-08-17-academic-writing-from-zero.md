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
current_block: "1E"
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
| AW-16 | `question → claim → evidence → reasoning` 最小链 | 2026-08-17 Block 1D：+3 结构复现；但加入材料没有的监管/保护与概率信息 | 第 +7 教学块 | 已排期 |
| AW-01 | 区分 topic 与题目要求的 task | 2026-08-17 Block 1C：知道题目要求判断移民的作用，但 topic 过窄，且把 `fiscal` 误读为“社会” | Block 1E（纠正后 +1） | 正在复习 |
| AW-01 | 识别题目 parts 与 restrictions | 2026-08-17 Block 1C：两个 parts、UK 与未来二十年正确；遗漏 fiscal problem scope | Block 1E（纠正后 +1） | 正在复习 |
| AW-01 | 处理 definitions 与 problematic vocabulary | 2026-08-17 Block 1D：正确区分 prevalence 与 intensity；`cause for concern` 的标准仍过宽 | Block 1E（换词 +1） | 正在复习 |
| AW-01 | 形成 provisional view 与 possible structure | 正在教学 | Block 1E 即时检查 | 正在教学 |

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

## Block 1A 讲评

这次核心链已经成立：

- **Claim**：提高女性比例本身可能不足以消除性别不平等。
- **Evidence**：你准确使用了“女性占比超过 40% 的公司晋升差距并未更小”，并补充了公开晋升标准与较小差距的发现。
- **Reasoning**：你解释了单纯改变人数结构未必足够，制度规则可能更直接关联晋升结果。

需要收紧的一处是：`女性占比可能对于晋升的影响很小` 超出了这份假设研究能证明的范围。研究只显示占比更高的公司没有更小的晋升差距，不能证明占比对所有晋升结果“影响很小”，也不能单凭相关性确定因果。更忠实的判断是：`提高女性占比本身可能不足以缩小晋升差距。`

## Block 1B｜Topic 不等于 Task

写 essay 时，先分清两件事：

- **Topic**：文章谈论的领域，例如 climate policy。
- **Task**：题目要求你对这个领域作出什么判断，例如“净零目标能否证明政策有效”。

只围绕 topic 收集信息，会写成“我知道什么”。学术写作要服从 task，写成“我根据什么证据作出什么判断”。这对应课件 question analysis 的第一个问题：**What do you have to do?**

> [!example] Social Media
> Topic 是 social media use；task 不是介绍平台或用户数量，而是判断大量使用本身是否足以构成 concern。

## 混合检查 1B｜新知识 + 1A 复习

> [!question] 教学情境（假设数据）
> **题目：**越来越多政府宣布 net-zero targets（净零排放目标）。这是否足以证明 climate policy 有效？
>
> 一项假设比较发现：没有强制措施的国家在宣布目标后，排放趋势变化很小；同时采用 carbon pricing（碳定价）或强制排放标准的国家，减排幅度更大。

请写 **四句中文**：

1. 这道题的 **task** 是要求你判断什么？
2. 给出你的直接答案，也就是 claim。
3. 准确使用一项或两项 evidence。
4. 解释 evidence 为什么支持 claim。

这四句是一个完整的小判断，不需要写背景介绍，也暂时不用英文。

## Block 1B 讲评

你正确识别了 task：判断“政府宣布净零目标”能否证明 climate policy 有效，而不是介绍各国气候政策。你也再次完成了 claim、evidence 与 reasoning 的结构。

这次有两处 evidence 边界需要收紧：

- `完全取决于` 太强。材料只显示执行措施与结果有关，不能证明政策效果完全由这一项决定。
- `更强硬的措施可能会导致` 带入了因果。这个比较只告诉我们两类国家的结果不同，未证明措施本身造成了全部差异。

更贴合材料的表达是：`宣布净零目标本身不足以证明政策有效；有执行工具的政策与更大减排相联系，因此判断有效性时必须检查目标是否落实为具体措施。`

## Block 1C｜题目有几部分？范围在哪里？

确认 task 后，再检查两件事：

1. **Parts**：题目要求完成几个工作？每一部分都必须回答，但不一定同等重要。
2. **Restrictions**：题目把回答限制在哪个时间、地点、群体、案例或 perspective（观察角度）？

> [!example] Social Media
> `Critically evaluate the benefits and harms of social media for adolescents in the UK since 2020.`
>
> Parts 是评价 benefits 和 harms；restrictions 是 adolescents、UK、since 2020。文章若改写全球所有年龄群体，就算内容正确，也没有完成这道题。

范围的作用不是让文章变窄而已，而是决定什么 evidence 有资格进入答案。研究对象、地点或时期不匹配时，“相关”不等于“可用”。

## 混合检查 1C｜Parts / restrictions + topic/task 复习

> [!question]
> `To what extent can immigration reduce the fiscal pressures created by population ageing in the UK over the next twenty years, and what limitations should policymakers consider?`

请用三条简短中文回答：

1. **Topic** 是什么？真正的 **task** 是什么？
2. 题目包含哪两个必须回答的 parts？哪个是中心判断？
3. 题目规定了哪些 restrictions？至少指出地点、时间和问题范围。

暂时不要回答移民究竟有没有效；这一轮只练把题目读完整。

## Block 1C 讲评

这次已经识别出两个 parts，也抓到了地点 `UK` 和时间 `over the next twenty years`。第一部分确实是中心判断，第二部分不是附加背景，而是必须回答、并用来限定第一部分的内容。

需要修正的关键词是 `fiscal`。它不是“社会的”，而是“财政的”：这里指人口老龄化给公共财政带来的压力，例如税收基础、养老金以及医疗和照护支出。把它读成 social pressures，会让后面寻找的 evidence 整体偏题。

完整分析应是：

- **Topic**：移民能否缓解英国人口老龄化带来的财政压力。
- **Task**：评价移民能缓解到什么程度，而不是只解释“怎样缓解”。
- **Parts**：判断缓解程度；分析政策制定者必须考虑的限制。第一项是中心判断，但两项都必须完成。
- **Restrictions**：英国、未来二十年、人口老龄化造成的财政压力。

## Block 1D｜先定义会改变答案的词

审题时，不需要把每个名词都写成字典定义。只需要处理两类会改变论证方向的词：

1. **Terms needing definition**：如果不先说明含义，读者不知道你在衡量什么。例如 `fiscal pressures` 应限定为公共收入与养老金、医疗和照护等支出之间的压力。
2. **Problematic vocabulary**：可能有多种合理解释的词。例如 `reduce` 可以指短期缓解，也可以指长期解决；两种解释会产生不同答案。

定义的作用是固定本篇文章的判断标准。它不能只是抄词典，也不能在写到一半时偷偷换意思。

## 混合检查 1D｜定义与歧义 + 最小论证链复习

> [!question] 教学情境（假设数据）
> **题目：**`More than 90% of adolescents in the UK use social media. Is this cause for concern?`
>
> 一项对 5,000 名英国青少年的假设研究发现：总体使用时长与较低 wellbeing（身心健康）的平均关联很弱；较大的负面关联主要集中在遭遇网络骚扰或强迫性使用的人群。

请简短完成三项：

1. 题目中的 `this` 至少有哪两种合理解释？
2. 在你的答案里，什么情况才算 `cause for concern`？给出一个可用于判断 evidence 的定义。
3. 写三句中文：直接 claim、准确 evidence、reasoning。不要把“很多人使用”自动当成“很多人受到伤害”。

## Block 1D 讲评

`this` 的歧义识别正确：它可以指 **prevalence**（有多少青少年使用），也可以指 **intensity**（每个青少年使用多少）。这是本轮最重要的审题判断。

`cause for concern` 的定义还需要变成可操作标准。`带来严重社会问题` 太宽，`对某些人有一些坏处` 又太低：几乎任何活动都可能伤害少数人。一个能指导 evidence 判断的版本是：`有证据显示，一群数量不可忽略的人受到实质伤害，或者一个可识别的高风险群体受到严重伤害。`

claim–evidence–reasoning 的结构再次成立，但材料边界有三处变化：

- 材料没有提到监管或网络保护，因此不能把它们写进本题 claim。
- 材料只说负面关联较大，没有给出“有不小概率”。
- `平均关联很弱` 不等于“没有明显证据”或“没有害处”。

贴合材料的版本是：`超过 90% 的青少年使用社交媒体，这一事实本身不足以构成普遍担忧。研究发现，总体使用时长与较低身心健康的平均关联很弱，而较大的负面关联集中于遭遇网络骚扰或强迫性使用者。因此，担忧应针对特定使用条件和高风险群体，而不是使用人数本身。`

## Block 1E｜先有暂定答案，再决定文章路线

完成 task、parts、restrictions 和 definitions 后，question analysis 还有两个问题：

1. **Do you have a view?** 先给一个 provisional view（暂定答案）。它可以在阅读后改变，但现在必须直接回应 task。
2. **How might you structure a response?** 给正文安排 paragraph jobs：每段要替总答案证明什么，而不是先决定要写哪位作者。

例如，净零题的暂定答案可以是“宣布目标本身不足以证明政策有效，必须检查具体执行工具”。相应段落工作可以是：先判断目标声明本身的证据，再比较执行工具，最后处理限制。答案决定结构，不是资料出现的顺序决定结构。

## 混合检查 1E｜完整审题 + 暂定路线

> [!question]
> `To what extent can gender quotas improve women's representation in senior leadership, and what limitations should organisations consider?`

暂时不需要查资料。请用三组短点回答：

1. **完整审题**：topic、真正的 task、两个 parts、restrictions，以及一个必须定义或可能有歧义的词。
2. **Provisional view**：用一句中文直接回答题目；它可以是暂定的，但不能只说“本文将讨论”。
3. **Paragraph jobs**：给正文安排两个或三个工作。只写每部分必须证明什么，不写作者名字。

<!-- student-os:learning-log:start -->

### 2026-08-17 16:54 BST

- **学习者输入**：明确纠正上一版：把自己当作没上过 Academic Writing 的小白；根据仓库笔记重新上完整课程；先教一部分再考一部分，并定期抽查很早以前的内容。
- **AI teaching/feedback**：承认旧目标错误，停用 essay-first 记录。建立 foundation course、review queue 与 `+1/+3/+7` 复习节奏。本轮教学 Academic Writing 最小模型，并给出一题三句构造检查；没有假设任何既有掌握。
- **当前 skill-map 位置**：Block 1A；`AW-16` 的最小 top-down reasoning chain。尚无本课程新证据。
- **单一下一动作**：学习者根据 Gender 假设研究写三句中文，分别承担 claim、evidence 与 reasoning。
- **internal continuation decision**：若三部分清楚且 evidence 未被改写，记录首次教学证据并把 AW-16 放入 +1/+3/+7 queue；下一块教学 topic 与 task 的区别，并混入一道 1A 旧题。若缺一部分，用 Climate Change 对比讲清后换材料重测，不要求重复原句。

### 2026-08-17 17:21 BST

- **学习者输入**：`提高女性员工比例能从被认为可能不能削减性别不平等. 一项调查指出,女性比例超过40%的公司晋升差距不比其他公司小. 这暗示了女性在公司中的占比可能对于晋升的影响很小.另外,该研究也指出公开晋升标准的部门晋升差距小,从中也许可以推断出公开晋升标准有助于在晋升中的性别平等`
- **AI teaching/feedback**：确认 claim、两项 evidence 与 reasoning 均已出现，Block 1A 的目标在明确结构支持下成立。指出唯一需要当场修正的 meaning 边界：观察到占比更高的公司没有更小晋升差距，只支持“提高占比本身可能不足”，不支持“占比对晋升影响很小”这一更宽且近似因果的判断。未要求重写原答案，直接进入 1B。
- **当前 skill-map 位置**：Block 1B；显式教学 `AW-01` 的 topic/task 区别，并在同一 Climate Change 题中执行 `AW-16` 的 +1 retrieval。
- **单一下一动作**：围绕 net-zero 假设研究写四句中文，依次完成 task、claim、evidence、reasoning。
- **internal continuation decision**：若能把 topic 与 task 分开，并再次完成准确小论证，则记录 AW-01 初次受支持证据，把 AW-16 排到 +3；下一块继续 question analysis 的 parts/restrictions，并在 review queue 到期时再抽查。若仍只介绍 climate policy 或改变 evidence scope，用一个 Demographics 对比讲清后换题重测。

### 2026-08-17 17:27 BST

- **学习者输入**：`给句话的,task要求我判断政府,很多政府或者越来越多政府,这个宣布零碳排放量计划,能否证明climate policy有效。 这是要求我判断的。我的claim是,这个完全取决于政府的作为方式和政府的选择的policy种类。因为有研究发现,没有强制措施的国家宣布目标后,他们排放趋势变化很小。采用碳定价或强制排放标准的国家减排幅度更大。那么这个就说明政府采取更强硬的措施,有可能会导致更好的减排效果。`
- **AI teaching/feedback**：确认 topic/task 区分正确，且在新材料中再次完成 claim–evidence–reasoning 结构。即时收紧两处会改变材料含义的问题：`完全取决于` 超出比较证据，`导致` 把相关性变成因果。给出忠实版本后不要求重写，进入 question analysis 的 parts/restrictions。
- **当前 skill-map 位置**：Block 1C；教学 `AW-01` 的 parts 与 restrictions，并执行 topic/task 的 +1 retrieval。`AW-16` 的结构证据排到 +3；strength/causality 风险保留到后续 AW-15/AW-17 正式教学。
- **单一下一动作**：分析 immigration/population-ageing 题目的 topic/task、两个 parts 与 place/time/problem restrictions。
- **internal continuation decision**：若识别完整，记录 AW-01 的第二个受支持应用，topic/task 排到 +3，并进入 definitions/problematic vocabulary；若遗漏 part 或把 restriction 当作背景，用一个 Social Media 对比后换题重测。

### 2026-08-17 17:35 BST

- **学习者输入**：`topic是population aging, task是移民能够减少多少的英国老龄化导致的社会压力。要求回答,怎么样减少,然后那个政策制定者应该考虑什么限制。中心部分应该是前一个,就是怎么样的减少,限制的是在英国,然后未来20年。`
- **AI teaching/feedback**：确认学习者识别了两个 parts、中心判断，以及 UK/未来二十年两项 restrictions。纠正一个会改变整篇 evidence selection 的语义错误：`fiscal pressures` 是人口老龄化造成的公共财政压力，不是“社会压力”；同时区分 `to what extent` 的程度判断与“怎么样减少”的机制描述，并补回遗漏的 problem-scope restriction。不要求重写原题，进入 definitions/problematic vocabulary。
- **当前 skill-map 位置**：Block 1D；继续 `AW-01` 的 definitions/problematic vocabulary，并在 Social Media 新题中复习 `AW-16` 的 +3 最小论证链。
- **单一下一动作**：解释 `this` 的两种读法，界定 `cause for concern`，再根据给定研究写 claim–evidence–reasoning 三句链。
- **internal continuation decision**：若定义能形成判断标准、歧义得到选择，且旧论证链不把 prevalence 改写成 harm，则记录 AW-01 当前应用与 AW-16 +3 证据，进入 question analysis 的 provisional view/possible structure；若定义仍只是同义替换或 evidence scope 被扩大，用 Climate Change 的 `effective` 对比讲清后换题重测。

### 2026-08-17 17:40 BST

- **学习者输入**：`第一题,this可能指的是要么是指用的人多,也可能是说年轻人用得多。第二个,带来严重的社会问题,或者对某些人会有一些坏处,could cause for concern。第三个,我的claim是,单纯的很多年轻人用并不能cause for concern,然后监管较少和对他们在网络上的保护较少,应该引起concern。我的evidence是那个研究发现,没有明显的证据显示使用时长和心理健康有明显的关联。但是,如果受到网络暴力或强迫性使用,有不小的概率会出现负面的状况,对于青少年来讲。这说明了单纯的使用时长不倾向于表现,不倾向于对年轻人的心理健康有害处。`
- **AI teaching/feedback**：确认已把 `this` 区分为使用人数/prevalence 与个人使用量/intensity，并再次形成 claim–evidence–reasoning。把 `cause for concern` 收紧为可用 evidence 判断的 harm threshold。指出三处 source-meaning 变化：无据加入监管/保护、无据加入概率、把 weak average association 改成 no clear evidence/无害；给出忠实对比后不要求重写原句。
- **当前 skill-map 位置**：Block 1E；教学 `AW-01` 的 provisional view/possible structure，并用 Gender 新题复测完整 question analysis。`AW-16` 的 +3 结构复现已记录，source fidelity 风险保留到 AW-15/AW-17 正式教学。
- **单一下一动作**：对 gender-quota 题完成完整审题，给出一句暂定答案，再安排两个或三个 paragraph jobs。
- **internal continuation decision**：若审题范围完整、暂定答案直接回应 extent/limitations、段落工作由答案驱动，则完成 AW-01 的本轮 supported chain 并进入 AW-02 thesis；若仍把 task 写成 mechanism 或定义不能影响 evidence selection，用 Climate Change 新题简短重测后再进入 thesis。

<!-- student-os:learning-log:end -->
