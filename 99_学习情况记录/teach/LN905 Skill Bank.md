---
student_os: ln905-skill-bank
title: LN905 Skill Bank
status: active
course: LN905
last_source_audit: 2026-08-13
neath_sync: synced
---

# LN905 Skill Bank

> [!summary] 用法
> 这是 LN905 唯一的跨会话技能状态源。课程笔记保存完整解释；[[99_学习情况记录/学习计划/LN905 Listening and Reading Practice|练习计划]]只保存 canonical tasks；每个 guided Markdown 只保存本次处方与逐轮证据。导师从这里选主技能和自然适用的静默 callback，助教不得自行换目标。

## 状态协议

- `new`：已从已上课程材料核验，但尚未留下完整的显式教学证据。
- `guided`：已经完成“解释 → 支架应用 → 变式应用 → 真实输出整合”。同日有提示的成功只到这里。
- `independent-1`：在新材料中第一次无提示正确调用。
- `stable`：在不同材料中两次无提示正确调用，而且至少一次来自 Friday/timed output。
- `repair`：在两个真正适用的情境中反复缺失或误用，需要重新刻意练习。
- callback 证据只写 `independent`、`guided`、`incorrect` 或 `not observable`；没有自然机会时只能写 `not observable`，不能算失败。
- 状态只能按 `new → guided → independent-1 → stable` 前进；任何阶段在两个适用情境反复失败可转 `repair`，修复后回 `guided`。

## 来源覆盖

完整文件清单由 [[07_Programme/01_LN905_LSE-language-class/LN905 PDF Contents|LN905 PDF Contents]] 保存；这里记录每组已学材料的提取结果，不按页码复制技能。

覆盖继承规则：清单中位于下列材料组内的每个文件都继承该行的“合并/排除/未激活”结果；今后 Moodle 新出现、尚未进入清单或无法判断是否已上课的文件保持待核验，不自动当作已学。

| 材料组 | 覆盖结果 | 处理 |
|---|---|---|
| Assessment Information 与 Paper A/B/C、Assignment marking criteria | 已提取任务约束、评分维度与 band 分水岭 | 合并到 `PA`、`PB`、`PC`、`AW` |
| Academic Writing Weeks 1–3 全部课件、workbook、sample essay | 已提取题目、thesis、结构、段落、Toulmin、source use、synthesis、voice、cohesion、revision | 合并到 `AW-01`–`AW-16` |
| Paper A introduction、AI workbook、Social Media/Gender practice 与 feedback | 已提取 selective notes、meaning/evidence map、summary、evaluation | 合并到 `PA-01`–`PA-05` |
| Paper B introduction、AI workbook、Social Media/Gender extracts 与 feedback | 已提取 question-led reading、source map、synthesis、citation | 合并到 `PB-01`–`PB-05` |
| Paper C presentation、discussion functions、Part 3、marking criteria | 已提取 presentation、contribution、floor、response | 合并到 `PC-01`–`PC-04` |
| Academic Interaction core、Gender、Demographics | 已提取 roles、co-construction、evaluation 与 speaking-to-writing | 合并到 `AI-01`–`AI-03` |
| Discussion & Debate Weeks 1–3 | 功能语言已提取；人物游戏、challenge cards、balloon/pyramid 场景本身不作为技能 | 合并到 `DISC-01`–`DISC-02`；课堂游戏明确排除 |
| Pronunciation sessions、transcript、TH materials 与 H5P | 已提取 intelligibility 所需的音位、schwa、connected speech、stress/chunking | 合并到 `PRON-01`–`PRON-03` |
| Social Media、Gender、Demographics topic input/listening/reading | 可迁移主题词已提取；逐题答案和一次性事实不作为技能 | 合并到 `LEX-02`–`LEX-04` 和词汇同步区 |
| Week 4 Climate Change 全部材料 | 文件已下载但尚未实际授课 | **未激活**：不建 skill 状态、不导入词汇；上课后增量提取 |
| Everyday English、课堂行政、课程游戏规则 | 与学术考核迁移无直接关系 | 明确排除 |
| *Academic Vocabulary in Use* | 仅作参考书 | 不批量导入；只收课堂实际教授、反复出现或真实输出需要的词 |
| LN905 Essay 原始论文 | 是练习/写作证据，不是技能课件 | 不从论文主题批量制造 skill；真实输出可作为 callback 证据 |

## Academic Writing

> [!important] 首批教学优先级
> `AW-16` 自上而下写作程序继续作为当前主轴；接下来的 Shared Writing 优先让 `AW-08` writer-led topic sentence、`AW-09` reporting noun/gerund、`AW-10` literature-as-evidence 和 `AW-11` evaluative language 分别完成一次显式闭环。不要在同一 part 同时把四项都当主技能。

### AW-01 · 把题目变成可执行任务
- 状态：`new`
- 触发/功能：拿到 essay、Paper B 或 assignment question 时，先确定 command、scope、concepts 与 source requirement。
- 为什么：不先读清任务，后面的好句子也可能回答错问题。
- 动作/框架：圈 command → 定义关键词 → 写限制范围 → 写一句“我必须证明什么”。
- 边界：不能偷换 command、扩大 population/time/context，或把讨论题写成描述题。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#二、先把题目变成写作任务|Academic Writing：读题]]；Assignment criteria。
- 适用：Paper B、Assignment；planning、introduction、whole essay。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 Shared Writing 开始前静默看是否先写任务要求。

### AW-02 · 控制性答案 / thesis
- 状态：`new`
- 触发/功能：材料很多但不知道全文到底要说什么时，形成可争辩、有限定的答案。
- 为什么：thesis 是选材和段落取舍的上位标准。
- 动作/框架：`Although X, this essay argues Y because A and B, within Z.` 可变形，不要求照抄。
- 边界：certainty、scope、causality 必须受来源支持；不能先列作者再拼答案。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#三、用 thesis 固定全文的答案|Academic Writing：thesis]]。
- 适用：Paper B、Assignment；thesis、introduction、conclusion。
- 证据/Callback：起点证据显示能形成 overall argument，但尚无系统四步闭环。
- 下一观察：Demographics Shared Writing 的 provisional answer。

### AW-03 · 选择宏观结构
- 状态：`new`
- 触发/功能：已有答案后，按论证逻辑选择 thematic、causal、problem–solution 或 comparative structure。
- 为什么：人类写作者先安排读者理解路径，再写句子。
- 动作/框架：把 thesis 拆成 2–4 个必须依次完成的 reasoning jobs。
- 边界：结构服务答案；不能因为来源有三篇就写成三个作者段。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#四、先搭论证结构，再写句子|Academic Writing：结构]]。
- 适用：Paper A/B、Assignment；outline、whole text。
- 证据/Callback：Listening teacher feedback 认可整体理解；Reading feedback 认可基本逻辑组织，但未完成结论。
- 下一观察：下一份 250+ 词 integrated output。

### AW-04 · 给每段一个 writer job
- 状态：`new`
- 触发/功能：从 thesis 进入 paragraph plan 时，先写该段对全文答案的功能。
- 为什么：段落不是材料容器，而是完成一个推理动作的小论证。
- 动作/框架：`段落功能 → writer claim → evidence role → warrant/implication → link back`。
- 边界：一个段落不能同时承担互不相干的多个 job。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#五、让每一段成为一个小论证|Academic Writing：段落]]。
- 适用：Paper A/B、Assignment；outline、paragraph。
- 证据/Callback：尚无显式闭环。
- 下一观察：AW-16 显式教学后的下一篇新材料。

### AW-05 · Claim–Evidence–Warrant 推理
- 状态：`new`
- 触发/功能：段落有 claim 和 citation，但读者仍不知道证据为什么支持判断时补 warrant。
- 为什么：引用不会自动完成推理。
- 动作/框架：先标 claim/ground，再问“这项 evidence 允许我得出哪一步、为什么”。
- 边界：warrant 不能发明机制；qualifier/rebuttal 不能被省略成绝对结论。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#六、用 Toulmin 检查论证能否站住|Academic Writing：Toulmin]]。
- 适用：Paper A/B、Assignment；analysis、evaluation、paragraph development。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 evidence-role paragraph。

### AW-06 · Description → Analysis → Evaluation
- 状态：`new`
- 触发/功能：写完 source content 后决定它意味着什么、为什么重要、证据有多强。
- 为什么：考试要求的不只是复述。
- 动作/框架：`what/source says → relation/mechanism → significance/strength/limitation for my answer`。
- 边界：评价必须针对真实 evidence、method、scope 或 relation，不能只写空泛的 “not convincing”。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-2-Gender/Description vs Analysis vs Evaluation.pdf|Description vs Analysis vs Evaluation]]。
- 适用：Paper A/B、Assignment；critical summary、essay paragraph。
- 证据/Callback：教师反馈确认已经尝试 evaluation，但 Listening 中误判了 evidence composition。
- 下一观察：下一次 mixed-evidence evaluation。

### AW-07 · Source evaluation 与 intended use
- 状态：`new`
- 触发/功能：决定来源是否值得用、放在哪个 claim 下、承担什么证据角色。
- 为什么：可靠来源也不一定适合当前论点。
- 动作/框架：currency/relevance/authority/accuracy/purpose → precise intended use。
- 边界：不能用作者声望替代对 evidence 与 question fit 的判断。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#七、把资料变成自己的论证|Academic Writing：source use]]；Assignment criteria。
- 适用：Assignment、annotated bibliography；source selection、annotation。
- 证据/Callback：Annotated bibliography 已提交；教师反馈认可 sources 与 connections，但要求更准确 synthesis。
- 下一观察：下一次 essay source decision。

### AW-08 · Writer-led topic sentence
- 状态：`new`
- 触发/功能：段首准备写 `Author argues...` 时，先写自己的段落判断。
- 为什么：读者必须先看见本段如何推进 writer 的答案，文献随后才成为 evidence。
- 动作/框架：writer claim/topic sentence → `As/According to...` evidence → writer explanation。
- 边界：writer claim 仍须被文献支持；不能把来源观点冒充自己的原创事实。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-3-Demographics/Week 3 Lesson 4.pdf|Week 3 Lesson 4：Effective topic sentences]]。
- 适用：Paper B、Assignment；topic sentence、paragraph opening。
- 证据/Callback：Lesson 4 已上课；尚无系统显式闭环。
- 下一观察：下一次自然适用的 Shared Writing；近期没有合适 task 时才用按功能聚合的 clinic。

### AW-09 · Reporting noun / gerund 留出主句给自己
- 状态：`new`
- 触发/功能：句子只有 `Author suggests X`，没有 writer comment 时重构句法。
- 为什么：主句的谓语位置应承担 significance、strength、limitation 或 relation。
- 动作/框架：`The suggestion that X ... is important/limited because...`；`Author's explanation of X...`；`By arguing X, Author...`。
- 边界：名词化或 gerund 不能改变原作者 certainty、scope、claim ownership。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-3-Demographics/Week 3 Lesson 4.pdf|Week 3 Lesson 4：Using verbs to present your voice]]。
- 适用：Paper A/B、Assignment；source integration、evaluation sentence。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次自然适用的 Shared Writing；近期没有合适 task 时才用按功能聚合的 clinic。

### AW-10 · `As Author argues...` 与 literature-as-evidence
- 状态：`new`
- 触发/功能：writer 已有 claim，需要把文献降为支持性从句或 citation 时调用。
- 为什么：句法层级让“这是我的判断；这是支持我的文献”清楚可见。
- 动作/框架：`As Author (Year) argues, [writer proposition].`；`[Writer claim] (Author, Year).`
- 边界：`As` 从句只适合文献确实支持 writer proposition；不得掩盖 disagreement 或 source limitation。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-3-Demographics/Week 3 Lesson 4.pdf|Week 3 Lesson 4：Using literature to support claims]]。
- 适用：Paper B、Assignment；claim support、synthesis。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次自然适用的 Shared Writing；近期没有合适 task 时才用按功能聚合的 clinic。

### AW-11 · 精确 evaluative language
- 状态：`new`
- 触发/功能：需要评价 source strength、limitation、relevance 或 originality 时，用有对象的学术评价词。
- 为什么：`good/bad` 不说明判断标准，也无法推进论证。
- 动作/框架：`convincing/persuasive evidence`、`limited/narrow definition`、`systematic investigation`、`overlooks/fails to account for...`。
- 边界：评价词必须有可观察依据；`systematic` 与 `systemic`、`comprehensive` 与 `wide-ranging` 不可随意互换。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-3-Demographics/Week 3 Lesson 4.pdf|Week 3 Lesson 4：Evaluative adjectives]]。
- 适用：Paper A/B、Assignment；evaluation、literature review。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次自然适用的 Shared Writing；近期没有合适 task 时才用按功能聚合的 clinic，之后再静默观察 lexical choice。

### AW-12 · Cohesion 与 reference chains
- 状态：`new`
- 触发/功能：段落关系不清或连接词堆积时，先确定真实逻辑关系再表达。
- 为什么：cohesion 来自 proposition 的连续发展，不是连接词数量。
- 动作/框架：relation → connector/reference chain/given-to-new order。
- 边界：不能用 `therefore` 制造不存在的因果，也不能用模糊 `this` 隐藏指代对象。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#九、Cohesion 是关系清楚，不是连接词越多越好|Academic Writing：cohesion]]。
- 适用：Paper A/B、Assignment；paragraph、whole-text flow。
- 证据/Callback：教师反馈认可整体逻辑，但要求更清楚说明 sources 如何共同工作。
- 下一观察：下一份 synthesis paragraph。

### AW-13 · Introduction–Conclusion 对齐
- 状态：`new`
- 触发/功能：开头承诺了答案后，在 conclusion 回答同一问题并收束 implication。
- 为什么：结尾证明全文完成了任务，而不是机械重复。
- 动作/框架：restate answer at higher level → synthesise reasons → implication；不加新 evidence。
- 边界：不能引入新论点；不能因时间不足省略正式任务的 conclusion。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#十、Introduction 与 conclusion 要回答同一个问题|Academic Writing：introduction/conclusion]]。
- 适用：Paper B、Assignment；introduction、conclusion、timed completion。
- 证据/Callback：Reading teacher feedback：文章略低于最低字数并未完成 conclusion。
- 下一观察：下一份 Friday/timed Paper B。

### AW-14 · 按论证影响排序修改
- 状态：`new`
- 触发/功能：时间有限时，先修 task/claim/evidence/relation，再批量修语言。
- 为什么：细枝末节不能劫持主技能和整篇完成度。
- 动作/框架：task fulfilment → meaning/source accuracy → structure/reasoning → recurring language → isolated errors。
- 边界：会改变 meaning、source ownership、certainty、scope、causality 的语言问题仍须立即修。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#十一、按“影响论证的程度”修改|Academic Writing：revision]]。
- 适用：Paper A/B、Assignment；revision、AI feedback。
- 证据/Callback：用户明确反馈旧 prompt 纠缠细枝末节；系统规则已修正，技能尚待显式闭环。
- 下一观察：下一次 guided output 的反馈顺序。

### AW-15 · Epistemic fidelity（legacy `EP-01`）
- 状态：`guided`
- 触发/功能：来源表达 probability、forecast、may/tend 或有限范围时，保留其 epistemic status。
- 为什么：把预测写成事实会改变 source meaning。
- 动作/框架：先标 certainty/scope → 选匹配 reporting/hedging → 完句后反查是否变强。
- 边界：不得把 `may / 80% probability / forecast` 写成无条件 `will/is`。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#第三步：先定 thesis，再给每段一个 job|Reading into Writing]]；Paper B criteria。
- 适用：Paper A/B、Assignment；paraphrase、summary、evaluation。
- 证据/Callback：2026-08-11 Demographics Reading：逐步提示后把 80% probability 放回预测句；`guided`，无独立证据。
- 下一观察：2026-08-14 Friday timed output；测试前不提醒。

### AW-16 · 自上而下写作程序（legacy `W-01`）
- 状态：`new`
- 触发/功能：写作开始时先完成 `题目要求 → 暂定答案 → 段落功能 → evidence role` 再写 prose。
- 为什么：避免从某个来源或某个漂亮句子出发，最后才发现全文没有答案。
- 动作/框架：task → provisional answer → paragraph jobs → evidence roles/selection → prose。
- 边界：不能把完整 source extraction 当作 writer planning；短材料只为写作决策服务时不能冒充 Reading practice。
- 来源：[[99_学习情况记录/teach/LN905 Exam Playbook|LN905 Exam Playbook]]；[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/04_Academic Writing#四、先搭论证结构，再写句子|Academic Writing：plan]]。
- 适用：Paper A/B、Assignment；planning、integrated writing。
- 证据/Callback：原账本只记录“2026-08-12 Demographics Shared Writing 计划首次示范”，没有实际四步证据，因此合法迁移为 `new`。
- 下一观察：先完成显式闭环；之后再看 2026-08-14 或下一次 Friday output。

## Paper A · Listening into Writing

### PA-01 · 听前预测与纸面筛选器
- 状态：`new`
- 触发/功能：播放前依据 task 预判要听的 claim、blocks、evidence 与 qualifiers。
- 为什么：一次播放中不能平均记录所有声音。
- 动作/框架：task keywords → 预留 claim/blocks/evidence/limits 区域 → 只把预测当假设。
- 边界：预测不得冒充听到的内容。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/01_Listening into Writing#听前：把纸设计成筛选器|Listening：听前]]。
- 适用：Paper A；pre-listening、notes。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 Listening input。

### PA-02 · 结构信号与选择性笔记
- 状态：`new`
- 触发/功能：听中用 signposts 区分 central claim、support、example、contrast、qualification、conclusion。
- 为什么：笔记的价值在于恢复 argument，不在于逐句抄写。
- 动作/框架：structure signal → block label → minimal keywords/arrows → evidence type。
- 边界：不能把 example 升格为 main claim，也不能漏掉转折后的修正。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/01_Listening into Writing#听中：按层级选择，而不是逐句追赶|Listening：听中]]。
- 适用：Paper A；listening notes。
- 证据/Callback：用户能抓 main argument 和 key concepts；mixed evidence 分类仍弱。
- 下一观察：下一次完整短演讲。

### PA-03 · Meaning/evidence map 与 source ownership
- 状态：`new`
- 触发/功能：听后把笔记恢复成 claim、supporting blocks、evidence type、warrant、scope。
- 为什么：没有 map 就不能准确 summary 或 evaluation。
- 动作/框架：speaker claim → block function → evidence owner/type → warrant → qualifier。
- 边界：personal experience、research、data 必须分别识别；不完整 evidence map 不进入评价。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/01_Listening into Writing#听后：先把笔记变成地图|Listening：听后]]。
- 适用：Paper A；notes-to-map、pre-writing。
- 证据/Callback：教师反馈：main argument 理解清楚，但把 personal experience 当成全部 evidence，漏掉 research/data。
- 下一观察：Gender feedback repair 或下一次 mixed-evidence talk。

### PA-04 · Critical summary 的组织选择
- 状态：`new`
- 触发/功能：map 可靠后，在 linear/thematic 结构中选择最能保留 speaker argument 的组织。
- 为什么：summary 必须压缩而不破坏论证结构。
- 动作/框架：central answer → 2–3 blocks → evidence/qualification → integrated comment。
- 边界：不能逐点堆 notes；不能把自己的评价混成 speaker claim。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/01_Listening into Writing#写作：先报告，再评价|Listening：写作]]。
- 适用：Paper A；200–400 word critical summary。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一份 Paper A partial/timed output。

### PA-05 · 针对 evidence 的评价
- 状态：`new`
- 触发/功能：评价 speaker argument 时，先承认 evidence composition，再判断 relevance、quality、scope 与 inference。
- 为什么：批错 evidence 会使评价本身失真。
- 动作/框架：`The speaker uses A/B/C; this supports X because...; however, it remains limited to Y...`。
- 边界：不能只凭 personal relevance 判断证据强弱，也不能忽略研究/data。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/01_Listening into Writing#Criticality 的五层问题|Listening：criticality]]；Paper A criteria。
- 适用：Paper A；critical comment。
- 证据/Callback：用户已经尝试 critical evaluation；准确性不足。
- 下一观察：完成 PA-03 后的同材料 Writing part。

## Paper B · Reading into Writing

### PB-01 · Question-led source reading
- 状态：`new`
- 触发/功能：读三篇 extracts 前，用 question 决定搜什么而不是平均精读。
- 为什么：限时阅读需要围绕 writer decision 获取信息。
- 动作/框架：question dimensions → source audit → skimming → selective close reading。
- 边界：不能只因关键词相同就认定段落相关。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#第一步：在读文本前拆 question|Reading：question]]。
- 适用：Paper B；pre-reading、source selection。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 genuine Reading input。

### PB-02 · 三源 meaning/evidence map
- 状态：`new`
- 触发/功能：把每篇 source 压成 claim、evidence、mechanism、scope 与 intended role。
- 为什么：写作需要可比较的 proposition，不是三份摘要。
- 动作/框架：每源一行 `claim | evidence | mechanism | scope | use`，再标 agreement/tension/complement。
- 边界：不能丢 source ownership、certainty 或研究范围。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#第二步：把三篇材料压成一张 source map|Reading：source map]]。
- 适用：Paper B；reading notes、pre-writing。
- 证据/Callback：教师反馈认可使用全部三源并连接 ideas，但要求更准确说明 sources 如何共同工作。
- 下一观察：下一套正式量级三 extracts。

### PB-03 · Writer-led synthesis
- 状态：`new`
- 触发/功能：多个来源共同支持、解释、限制或反驳 writer claim 时，把关系写在同一段里。
- 为什么：synthesis 是围绕问题组织 sources，不是按作者轮流汇报。
- 动作/框架：writer claim → Source A role → relation to B/C → warrant/limitation → answer link。
- 边界：不能虚构 agreement；并列 citation 不自动等于 synthesis。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#第四步：让 synthesis 在段落里可见|Reading：synthesis]]。
- 适用：Paper B、Assignment；synthesis paragraph。
- 证据/Callback：起点强项是连接来源；尚未完成系统显式闭环。
- 下一观察：Demographics synthesis task。

### PB-04 · Quote、paraphrase、summary 与 citation 分工
- 状态：`new`
- 触发/功能：根据证据角色选择直接引语、准确改写或压缩总结，并明确 primary/secondary ownership。
- 为什么：不同转换服务不同写作目的。
- 动作/框架：先定 intended use → 选 transformation → 保留 proposition/certainty/scope → cite。
- 边界：换词不等于 paraphrase；citation 不能修复 meaning distortion 或 patchwriting。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#Quote、paraphrase、summary 的分工|Reading：source transformation]]。
- 适用：Paper B、Assignment；evidence sentence、reference。
- 证据/Callback：教师认可有 paraphrase attempt；准确性仍需结合 AW-15 观察。
- 下一观察：下一份完整 paragraph 后批量核验。

### PB-05 · Figure description 与 implication 分开
- 状态：`new`
- 触发/功能：遇到 graph/table 时先写可见 pattern，再判断它对 question 的意义。
- 为什么：数据本身与 writer inference 是两个推理层。
- 动作/框架：description → comparison/trend → limitation → implication for claim。
- 边界：图表只能支持其 measures/sample/time 范围；不能把 correlation 写成 causation。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/03_Reading into Writing#Figure reading：description 与 implication 分开|Reading：figures]]。
- 适用：Paper B、Academic Interaction；data commentary。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一份含 figure 的材料。

## Paper C、Academic Interaction 与 Discussion

### PC-01 · 两分钟 presentation 结构
- 状态：`new`
- 触发/功能：短时间口头回答时选择 problem–solution、chronological 或 thematic path，并明确 signposting。
- 为什么：听众不能回读，需要更明显的结构。
- 动作/框架：answer/opening → 2–3 points → signposts → concise close。
- 边界：结构必须适合 prompt；delivery 技巧不能替代 content。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#两分钟 presentation：先选结构|Speaking：presentation]]。
- 适用：Paper C；individual presentation。
- 证据/Callback：课堂材料已学；尚无显式闭环。
- 下一观察：Paper C skill clinic。

### PC-02 · 高质量 contribution 的最小单位
- 状态：`new`
- 触发/功能：讨论中贡献一个可被他人回应的 claim，而不是只表态。
- 为什么：学术讨论评估 reasoning 与 interaction。
- 动作/框架：claim → reason/evidence → link to prompt/previous speaker → invite response。
- 边界：不能垄断 floor；evidence/source ownership 仍须准确。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#一次高质量 contribution 的最小单位|Speaking：contribution]]。
- 适用：Paper C、Academic Interaction；discussion turn。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

### PC-03 · Gain、maintain、yield the floor
- 状态：`new`
- 触发/功能：自然进入、保持或交还发言权。
- 为什么：内容只有被合适地放入共同对话才算 interaction。
- 动作/框架：entry signal → concise point → link/hand over。
- 边界：不以抢话、超长发言或机械套句破坏互动。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#Gain、maintain、yield the floor|Speaking：floor]]。
- 适用：Paper C、Academic Interaction；turn management。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

### PC-04 · Listen–Build–Challenge–Repair
- 状态：`new`
- 触发/功能：回应他人时准确复述其 point，再补充、限制、挑战或澄清。
- 为什么：Paper C 的 listening 证据体现在回应质量，而非沉默听完。
- 动作/框架：brief uptake → relation → own reason/evidence → return to group。
- 边界：不能把对方的话夸大后再反驳；challenge 针对 idea，不针对 person。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#口头功能库：按作用记，不按句子背|Speaking：functions]]；Paper C criteria。
- 适用：Paper C；discussion response。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

### AI-01 · Roles 与 co-construction
- 状态：`new`
- 触发/功能：Academic Interaction 中用 discussion leader、summariser、evidence checker 等角色共同完成理解。
- 为什么：目标是共同建构而不是轮流汇报个人答案。
- 动作/框架：role action → reference source/peer → connect ideas → shared conclusion/open question。
- 边界：角色不等于固定台词；个人贡献必须回应共同任务。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#Academic Interaction：共同建构，不是轮流汇报|Speaking：Academic Interaction]]。
- 适用：Academic Interaction、Paper C；group work。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

### AI-02 · Evaluator 五问
- 状态：`new`
- 触发/功能：评价讨论或来源时检查 claim、evidence、reasoning、scope 与 unanswered question。
- 为什么：把“我同不同意”转成可解释的学术判断。
- 动作/框架：What claim? What support? Why connected? Under what scope? What remains?
- 边界：缺少可观察 evidence 时记不确定，不能发明理由。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#Evaluator 的五问|Speaking：evaluator]]。
- 适用：Academic Interaction、Paper A/B/C；evaluation。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 discussion-to-writing extraction。

### AI-03 · Discussion → Writing extraction
- 状态：`new`
- 触发/功能：课后把口头 discussion 压成 claims、evidence、relations、open questions，供写作使用。
- 为什么：口头活动只有转化成可调用信息才进入共同 Writing 树。
- 动作/框架：speaker/claim → evidence → relation → usable paragraph role。
- 边界：不能把讨论中的未经支持意见升级为文献证据。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#从 speaking 转成 writing|Speaking：to writing]]。
- 适用：Academic Interaction、Assignment、Paper B planning。
- 证据/Callback：尚无显式闭环。
- 下一观察：下一次 Academic Interaction 课后。

### DISC-01 · Elicit、Clarify、Include
- 状态：`new`
- 触发/功能：信息不清或成员未进入讨论时，追问、澄清、邀请并核对理解。
- 为什么：高质量学术讨论需要让 reasoning 可见且让组员能够接续。
- 动作/框架：`Could you clarify...?`、`What makes you think...?`、`How does this relate to...?`、邀请具体成员。
- 边界：问题必须服务内容，不变成审问或空泛 “what do you think”。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Discussion-and-Debate/Week-1/Discussion Week 1 PPT.pdf|Discussion Week 1]]。
- 适用：Paper C、Academic Interaction；discussion management。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

### DISC-02 · Qualify disagreement 与 build consensus
- 状态：`new`
- 触发/功能：需要不同意、rebut 或收束分歧时，准确限定 disagreement 并提出共同点。
- 为什么：有效 challenge 推进论证，不只是礼貌替换。
- 动作/框架：acknowledge valid part → locate disagreement → reason/evidence → possible synthesis/decision。
- 边界：不能用 hedge 抹掉立场，也不能歪曲对方原 claim。
- 来源：[[07_Programme/01_LN905_LSE-language-class/01_LN905_note/02_Speaking into Writing#Discussion drills：把技巧练到可调用|Speaking：discussion drills]]。
- 适用：Paper C、Academic Interaction；challenge、rebuttal、consensus。
- 证据/Callback：尚无显式闭环。
- 下一观察：Paper C/Interaction skill clinic。

## Pronunciation for academic interaction

### PRON-01 · 音位辨认与影响 intelligibility 的目标音
- 状态：`new`
- 触发/功能：因音位混淆导致学术词或关系词难以理解时，用 phonemic map 与最小对立定位。
- 为什么：目标不是口音消除，而是让关键 content 可被听懂。
- 动作/框架：听辨 → 定位发音部位/voicing → 词中练习 → academic phrase 中复现。
- 边界：只有影响 intelligibility 或反复出现的音才进入训练；不逐词纠正口音。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Pronunciation/Interactive/Session 1 - The Sounds of English.pdf|Sounds of English]]；[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Pronunciation/th sounds.pdf|TH sounds]]。
- 适用：Paper C、Academic Interaction；spoken academic output。
- 证据/Callback：尚无显式闭环。
- 下一观察：Pronunciation skill clinic。

### PRON-02 · Schwa 与 connected speech
- 状态：`new`
- 触发/功能：功能词过重、每词分开或听不出自然语流时，练弱读、linking 与 reduction。
- 为什么：自然节奏同时改善 speaking fluency 与 lecture listening segmentation。
- 动作/框架：标 content/function words → 弱读非重读音节 → chunk 内连接 → 录音复听。
- 边界：清晰优先于速度；不能为了连读吞掉改变 meaning 的音。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Pronunciation/Interactive/Session 2 - The Schwa.pdf|Schwa]]；[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Pronunciation/Interactive/Session 3 - Connected Speech.pdf|Connected Speech]]。
- 适用：Paper C、Listening；delivery、segmentation。
- 证据/Callback：尚无显式闭环。
- 下一观察：Pronunciation skill clinic。

### PRON-03 · Sentence stress、chunking 与 stance
- 状态：`new`
- 触发/功能：口头 claim、contrast、qualification 不突出时，用 stress/chunking 显示信息层级。
- 为什么：听众靠重音和停顿恢复 argument structure。
- 动作/框架：标 focus word → 按 meaning units 切 chunk → stress contrast/qualification → 正常速度复现。
- 边界：不能每个词都重读；停顿位置不能割裂固定搭配或从句关系。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/07_Language-Skills/Pronunciation/Interactive/Session 4 - Sentence & Word Stress.pdf|Sentence & Word Stress]]。
- 适用：Paper C、Academic Interaction；presentation、discussion。
- 证据/Callback：尚无显式闭环。
- 下一观察：Pronunciation skill clinic 或下一次 presentation。

## Lexical access

### LEX-01 · Academic reporting、stance 与 evaluation
- 状态：`new`
- 触发/功能：需要准确表达 source action、certainty、relation、strength 或 limitation 时调用词族与搭配。
- 为什么：词汇必须服务信息关系，不能只是同义替换。
- 动作/框架：按功能检索 reporting verb、hedge、relation verb、evaluative adjective/collocation，再反查语义边界。
- 边界：`argue/suggest/report/demonstrate`、`systematic/systemic` 等不可当装饰性同义词轮换。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-3-Demographics/Week 3 Lesson 4.pdf|Week 3 Lesson 4]]；Weeks 1–3 Academic Writing。
- 适用：Paper A/B/C、Assignment；all academic outputs。
- 证据/Callback：教师反馈反复指出 academic word choice；尚无显式词汇闭环。
- 下一观察：每日词灵复现 + 后续真实 output。

### LEX-02 · Social Media topic vocabulary
- 状态：`new`
- 触发/功能：讨论 privacy、well-being、platform risks 与 regulation 时准确调用主题概念。
- 为什么：主题词不足会同时阻断 input comprehension 与 English production。
- 动作/框架：词灵 retrieval → collocation/example → source-map/claim 中自然调用。
- 边界：定义需区分 privacy、anonymity、surveillance、data protection；不背孤立中文对译。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/08_Topics/Week-1-Social-Media/Input-and-Discussion/Topic Input Social Media.pdf|Topic Input Social Media]]。
- 适用：Paper A/B/C；Social Media。
- 证据/Callback：Week 1 已上课；未记录系统显式闭环。
- 下一观察：词灵复现或跨主题 comparison。

### LEX-03 · Gender topic vocabulary
- 状态：`new`
- 触发/功能：讨论 stereotype、socialisation、privilege、feminism 与 equality 时准确表达机制和范围。
- 为什么：不认识关键词会导致听读材料的 argument blocks 无法恢复。
- 动作/框架：词灵 retrieval → collocation/example → evidence map/claim 中自然调用。
- 边界：个人经验、研究发现与 normative claim 必须分开；主题词不自动证明因果。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/08_Topics/Week-2-Gender/Listening/Gender - Listening Vocabulary.pdf|Gender Listening Vocabulary]]；Gender topic materials。
- 适用：Paper A/B/C；Gender。
- 证据/Callback：Week 2 已上课；旧练习显示 lexical access 阻塞理解，未完成系统闭环。
- 下一观察：词灵复现或 Friday callback。

### LEX-04 · Demographics topic vocabulary
- 状态：`new`
- 触发/功能：讨论 fertility、mortality、population growth/decline 与 ageing 时准确保留统计范围和趋势。
- 为什么：主题词与数量/趋势限定直接决定 summary accuracy。
- 动作/框架：词灵 retrieval → collocation/example → graph/source claim 中自然调用。
- 边界：prediction、observed trend 与 policy claim 必须分开；population growth 不能自动写成 overpopulation。
- 来源：[[07_Programme/01_LN905_LSE-language-class/PDF/08_Topics/Week-3-Demographics/Input-and-Discussion/Topic Input Demographics.pdf|Topic Input Demographics]]。
- 适用：Paper A/B/C；Demographics。
- 证据/Callback：Week 3 已上课；AW-15 暴露 prediction certainty 风险。
- 下一观察：每日词灵 + 2026-08-14 Friday outputs。

> [!warning] Climate Change 尚未激活
> 远端 `LN905 Climate Change` 词书可以存在为空，但 Week 4 实际上课前不创建 `LEX` skill、不导入 topic vocabulary，也不把已下载课件当作“已学”。

## Callback 与写回协议

1. 每个 guided part 只有一个主技能；导师可选择任意数量、但必须与当前材料自然适用且已到观察时机的旧技能作为静默 callback。
2. 第一次完整输出前不得透露 callback ID、功能提示或检查项。没有自然机会写 `not observable`。
3. 自主正确写 `independent`；出现机会但误用写 `incorrect`；给过任何提示后正确只写 `guided`。
4. 缺失时依次只用：功能提示 → 结构提示 → 简短对比示范。同一细节不连续纠缠，次要语言问题在完整输出后批量反馈。
5. Friday/timed output 测试前不提醒；测试后把证据写回对应 skill。一次自主正确只到 `independent-1`；第二次且包含 timed/Friday 才到 `stable`。
6. `new` 只有在解释、支架应用、变式应用、真实输出整合四步均有记录后才可变 `guided`。
7. 当日助教只能按 mentor brief 的 `技能状态写回规则` 追加本次证据，并对 brief 点名的 ID 执行合法状态转换；不得选新技能、改下一观察策略或把普通语言反馈写成 mastery evidence。

## 词汇同步区

- 本区是匿词/词灵同步的本地真源。句法框架保留在上面的技能记录，不伪装成单词。
- 同一 lemma/phrase 只出现一次；`topics` 可以有多个标签，`collection` 是唯一远端归属。
- 同步只读取和新增/更新本系统管理的词书与词条，不移动其他词书中的已有词，不提供删除路径。
- 远端失败不影响本地训练；把 frontmatter `neath_sync` 改为 `pending`，下一次 `/today` 重试。

<!-- student-os:neath-vocabulary:start -->
```json
{
  "collections": [
    {"name": "LN905 Academic Core", "description": "LN905 academic reporting, stance, source use and evaluation"},
    {"name": "LN905 Social Media", "description": "LN905 Week 1 Social Media vocabulary"},
    {"name": "LN905 Gender", "description": "LN905 Week 2 Gender vocabulary"},
    {"name": "LN905 Demographics", "description": "LN905 Week 3 Demographics vocabulary"},
    {"name": "LN905 Climate Change", "description": "LN905 Week 4 vocabulary; kept empty until the class is taught"}
  ],
  "entries": [
    {"word": "argue", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "提出并支持一个可争辩的立场；不等同于 report", "collocations": ["argue that", "argue for/against"], "sentence": "The authors argue that the observed association requires a causal explanation."},
    {"word": "suggest", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "提出可能的解释或较弱结论", "collocations": ["suggest that", "evidence suggests"], "sentence": "The evidence suggests that the effect may vary across groups."},
    {"word": "demonstrate", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "用充分证据显示；强度通常高于 suggest", "collocations": ["demonstrate that", "clearly demonstrate"], "sentence": "The experiment demonstrates that the intervention changed reported behaviour."},
    {"word": "report", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "陈述研究观察、数据或结果，不自动表示作者赞同", "collocations": ["report findings", "report that"], "sentence": "The study reports a decline in average participation."},
    {"word": "indicate", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "证据指向某个解释或意义", "collocations": ["indicate that", "results indicate"], "sentence": "The results indicate that context matters."},
    {"word": "illustrate", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "用例子或证据使观点可见；不等于因果证明", "collocations": ["illustrate how", "clearly illustrate"], "sentence": "The case illustrates how institutional rules shape individual choices."},
    {"word": "warrant", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "连接 evidence 与 claim 的推理依据", "collocations": ["justify the warrant", "underlying warrant"], "sentence": "The paragraph must explain the warrant connecting the evidence to its claim."},
    {"word": "qualifier", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "限定 claim 的 certainty、scope 或条件", "collocations": ["add a qualifier", "appropriate qualifier"], "sentence": "A qualifier prevents a probabilistic finding from becoming an absolute claim."},
    {"word": "rebuttal", "collection": "LN905 Academic Core", "topics": ["academic-core", "discussion"], "meaning": "回应反例、限制或反方理由", "collocations": ["offer a rebuttal", "address a rebuttal"], "sentence": "A strong rebuttal acknowledges the opposing reason before answering it."},
    {"word": "convincing evidence", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "在明确标准下具有说服力的证据", "collocations": ["provide convincing evidence"], "sentence": "The longitudinal data provide convincing evidence of a persistent association."},
    {"word": "systematic investigation", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "按有组织、可说明的方法进行的调查；不是 systemic", "collocations": ["conduct a systematic investigation"], "sentence": "The researchers conducted a systematic investigation of the policy outcomes."},
    {"word": "limited in scope", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "证据或结论的适用范围有限", "collocations": ["remain limited in scope"], "sentence": "The study is informative but limited in scope because it examines one city."},
    {"word": "overlook", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "未考虑一个与结论有关的重要因素", "collocations": ["overlook the role of", "overlook the fact that"], "sentence": "The model overlooks the role of informal care."},
    {"word": "fail to account for", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "没有把关键机制、差异或证据纳入解释", "collocations": ["fail to account for variation"], "sentence": "The explanation fails to account for variation across age groups."},
    {"word": "provide a basis for", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "为后续判断或行动提供依据", "collocations": ["provide a basis for comparison"], "sentence": "These findings provide a basis for a more targeted policy response."},
    {"word": "highlight the need to", "collection": "LN905 Academic Core", "topics": ["academic-core"], "meaning": "显示采取某项研究或行动的必要性", "collocations": ["highlight the need to examine"], "sentence": "The conflicting results highlight the need to examine contextual differences."},
    {"word": "data privacy", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "个人数据的收集、使用、共享与保护条件", "collocations": ["data privacy concerns", "protect data privacy"], "sentence": "Data privacy concerns arise when platforms collect information without meaningful consent."},
    {"word": "targeted advertising", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "基于用户数据定向投放的广告", "collocations": ["targeted advertising practices"], "sentence": "Targeted advertising relies on detailed user profiles."},
    {"word": "online surveillance", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "企业或政府对线上活动的持续追踪", "collocations": ["protection from online surveillance"], "sentence": "Users may define privacy as protection from online surveillance."},
    {"word": "misinformation", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "错误或误导性信息，不必然包含故意欺骗", "collocations": ["spread misinformation", "counter misinformation"], "sentence": "Platform design can accelerate the spread of misinformation."},
    {"word": "attention span", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "持续集中注意力的能力或时长", "collocations": ["reduce attention span"], "sentence": "Frequent interruptions may reduce users' attention span."},
    {"word": "fear of missing out", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "担心错过他人活动或信息的焦虑", "collocations": ["experience fear of missing out"], "sentence": "Fear of missing out can encourage repeated checking."},
    {"word": "seeking validation", "collection": "LN905 Social Media", "topics": ["social-media"], "meaning": "通过他人反应寻求认可", "collocations": ["engage in validation-seeking behaviour"], "sentence": "Seeking validation may make users more sensitive to visible feedback metrics."},
    {"word": "gender expectations", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "社会对不同 gender 应如何表现的期待", "collocations": ["challenge gender expectations"], "sentence": "Gender expectations can restrict individual choices."},
    {"word": "socialisation", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "个体学习社会规范、角色与行为的过程", "collocations": ["gender socialisation", "process of socialisation"], "sentence": "Socialisation can exaggerate perceived differences between groups."},
    {"word": "self-fulfilling process", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "预期改变行为，进而使原预期看似真实的过程", "collocations": ["become a self-fulfilling process"], "sentence": "Repeated expectations may become a self-fulfilling process."},
    {"word": "fragile ego", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "容易因身份威胁而受损的自我感", "collocations": ["protect a fragile ego"], "sentence": "The speaker links rigid masculinity to a fragile ego."},
    {"word": "breadwinner", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "家庭主要经济供养者", "collocations": ["primary breadwinner"], "sentence": "Traditional expectations often position men as the primary breadwinner."},
    {"word": "emasculated", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "感到男性身份或权力被削弱", "collocations": ["feel emasculated"], "sentence": "Some men may feel emasculated when gender roles change."},
    {"word": "unsolicited advice", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "未经请求而给出的建议", "collocations": ["give unsolicited advice"], "sentence": "Unsolicited advice can reveal assumptions about authority."},
    {"word": "tongue-in-cheek", "collection": "LN905 Gender", "topics": ["gender"], "meaning": "以戏谑或反讽方式表达，并非字面认真", "collocations": ["make a tongue-in-cheek comment"], "sentence": "The remark was intended to be tongue-in-cheek."},
    {"word": "demography", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "对人类人口及其结构变化的统计研究", "collocations": ["study demography", "demographic analysis"], "sentence": "Demography examines changes in the size and structure of populations."},
    {"word": "overpopulation", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "人口规模相对于资源或承载能力过大", "collocations": ["fears of overpopulation"], "sentence": "Fears of overpopulation shaped earlier population-control campaigns."},
    {"word": "population control", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "旨在影响人口增长或规模的政策与措施", "collocations": ["population-control campaign"], "sentence": "Population-control policies can raise ethical and social questions."},
    {"word": "fertility rate", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "一定人群中生育水平的统计指标", "collocations": ["declining fertility rate"], "sentence": "The fertility rate has declined as family structures and health outcomes have changed."},
    {"word": "infant mortality", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "婴儿在一岁前死亡的统计水平", "collocations": ["reduce infant mortality"], "sentence": "Lower infant mortality can influence family-size decisions."},
    {"word": "maternal health", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "与怀孕、分娩及产后相关的女性健康", "collocations": ["improve maternal health"], "sentence": "Investment in maternal health can accompany demographic change."},
    {"word": "population decline", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "人口总量随时间减少", "collocations": ["experience population decline"], "sentence": "Some countries are more concerned about population decline than rapid growth."},
    {"word": "population implosion", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "对快速、持续人口收缩的强调性说法", "collocations": ["risk of population implosion"], "sentence": "The phrase population implosion frames demographic decline as an urgent threat."},
    {"word": "population projection", "collection": "LN905 Demographics", "topics": ["demographics", "academic-core"], "meaning": "基于假设和模型对未来人口的估计，不是确定事实", "collocations": ["long-term population projection"], "sentence": "A population projection should retain its assumptions and uncertainty."},
    {"word": "carrying capacity", "collection": "LN905 Demographics", "topics": ["demographics"], "meaning": "环境在既定条件下可持续支持的人口或生物数量", "collocations": ["planetary carrying capacity"], "sentence": "Debates about carrying capacity depend on assumptions about technology and consumption."}
  ]
}
```
<!-- student-os:neath-vocabulary:end -->

## 维护记录

- 2026-08-13：完成 Weeks 1–3 初次回溯；迁移 legacy `EP-01` → `AW-15`、`W-01` → `AW-16`；Week 4 保持未激活。
- 2026-08-13：首次建立 Academic Core、Social Media、Gender、Demographics 词汇清单；Climate Change 仅保留空集合。
